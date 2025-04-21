// Majestic RP Predictor с авторизацией пользователей (Firebase) и разграничением прав доступа
import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceArea, BarChart, Bar, Cell } from "recharts";
import { initializeApp } from "firebase/app";
import { getFirestore, collection, addDoc } from "firebase/firestore";
import { getAuth, onAuthStateChanged, signInWithEmailAndPassword, signOut } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyDao_tTRmfyXmmE2pQ9SyMVoF7BrmSUHcc",
  authDomain: "majpred.firebaseapp.com",
  projectId: "majpred",
  storageBucket: "majpred.firebasestorage.app",
  messagingSenderId: "143505491426",
  appId: "1:143505491426:web:213f6019ee3993a83a65bb",
  measurementId: "G-ZT2BKWCB0P"
};

const ADMIN_UID = "1lIJdWqSyBhVVj3TFDEf4YMgDjY2";

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const auth = getAuth(app);

const fieldToIndex = { "2x": 0, "3x": 1, "5x": 2, "10x": 3 };
const barColors = { "2x": "#A0A0A0", "3x": "#E74C3C", "5x": "#2ECC71", "10x": "#FFD700" };
const blockColors = { "2x": "#999", "3x/5x": "#8bc34a" };

export default function MajesticPredictor() {
  const [user, setUser] = useState(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [session, setSession] = useState([]);
  const [model, setModel] = useState(null);
  const [depth, setDepth] = useState(10);
  const [blocks, setBlocks] = useState([]);
  const [no5xCount, setNo5xCount] = useState(0);
  const [lastPrediction, setLastPrediction] = useState(null);
  const [nextPrediction, setNextPrediction] = useState(null);
  const [totalPredictions, setTotalPredictions] = useState(0);
  const [correctPredictions, setCorrectPredictions] = useState(0);
  const [bets, setBets] = useState({ "2x": 0, "3x": 0, "5x": 0, "10x": 0 });

  const isAdmin = user?.uid === ADMIN_UID;

  useEffect(() => {
    onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });
  }, []);

  useEffect(() => {
    if (!user) return;
    const loadOrCreateModel = async () => {
      try {
        const loadedModel = await tf.loadLayersModel("indexeddb://majestic-rp-model-5x");
        if (loadedModel.inputs[0].shape[1] !== depth + 6) throw new Error("Сброс модели");
        setModel(loadedModel);
      } catch {
        const newModel = tf.sequential();
        newModel.add(tf.layers.dense({ inputShape: [depth + 6], units: 32, activation: "relu" }));
        newModel.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
        newModel.compile({ optimizer: "adam", loss: "binaryCrossentropy" });
        setModel(newModel);
      }
    };
    loadOrCreateModel();
  }, [depth, user]);

  const saveToFirestore = async (data) => {
    if (!user) return;
    await addDoc(collection(db, "sessions"), {
      uid: user.uid,
      ...data,
      timestamp: new Date()
    });
  };

  const predictForward = async () => {
    if (!model || session.length < depth) return;
    const lastN = session.slice(-depth).map(f => fieldToIndex[f] / 3);
    const blockInput = 0.5;
    const betValues = ["2x", "3x", "5x", "10x"].map(k => Math.min(bets[k] / 100, 1));
    const input = [...lastN, blockInput, Math.min(no5xCount / 20, 1), ...betValues];
    const inputTensor = tf.tensor2d([input]);
    const result = model.predict(inputTensor);
    const value = (await result.data())[0];
    setNextPrediction(value);
  };

  const trainModel = async (inputSeq, is5x) => {
    if (!model || !isAdmin) return;
    const inputTensor = tf.tensor2d([inputSeq]);
    const outputTensor = tf.tensor2d([[is5x ? 1 : 0]]);
    await model.fit(inputTensor, outputTensor, { epochs: 3 });
    await model.save("indexeddb://majestic-rp-model-5x");
  };

  const addResult = async (selectedField) => {
    const newSession = [...session, selectedField];
    setSession(newSession);
    setNo5xCount(selectedField === "5x" ? 0 : no5xCount + 1);
    const lastN = newSession.slice(-depth).map(f => fieldToIndex[f] / 3);
    const blockInput = 0.5;
    const betValues = ["2x", "3x", "5x", "10x"].map(k => Math.min(bets[k] / 100, 1));
    if (lastN.length === depth) {
      const input = [...lastN, blockInput, Math.min(no5xCount / 20, 1), ...betValues];
      if (isAdmin) await trainModel(input, selectedField === "5x");
      await predictForward();
      await saveToFirestore({ session: newSession, bets });
    }
  };

  const signIn = () => {
    signInWithEmailAndPassword(auth, email, password).catch(console.error);
  };

  const signOutUser = () => {
    signOut(auth);
  };

  if (!user) {
    return (
      <div style={{ padding: 20 }}>
        <h2>🔐 Вход</h2>
        <input placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} />
        <input placeholder="Пароль" type="password" value={password} onChange={e => setPassword(e.target.value)} />
        <button onClick={signIn}>Войти</button>
      </div>
    );
  }

  const sessionData = session.map((f, i) => ({ name: `${i + 1}`, value: 1, color: barColors[f], label: f }));

  return (
    <div style={{ fontFamily: "sans-serif", padding: "1rem" }}>
      <h1>🎯 Majestic RP Predictor (с авторизацией)</h1>
      <p>👤 Пользователь: {user.email} <button onClick={signOutUser}>Выйти</button></p>
      <p>🔮 Следующий прогноз: {nextPrediction !== null ? `${(nextPrediction * 100).toFixed(1)}%` : "-"}</p>
      <p>🧠 Последний результат: {lastPrediction !== null ? `${(lastPrediction * 100).toFixed(1)}%` : "-"}</p>
      <div style={{ marginBottom: 20 }}>
        {["2x", "3x", "5x", "10x"].map(f => (
          <div key={f} style={{ display: 'inline-block', marginRight: 10 }}>
            <button onClick={() => addResult(f)} style={{ padding: '8px 12px', backgroundColor: barColors[f], color: '#fff', border: 'none', borderRadius: 4 }}>{f}</button>
            <input type="number" placeholder="ставка" value={bets[f]} onChange={e => setBets({ ...bets, [f]: Number(e.target.value) })} style={{ width: 60, marginLeft: 5 }} />
          </div>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={150}>
        <BarChart data={sessionData}>
          <XAxis dataKey="name" />
          <Tooltip formatter={(v, n, props) => props.payload.label} />
          <Bar dataKey="value">
            {sessionData.map((entry, i) => (
              <Cell key={`cell-session-${i}`} fill={entry.color} stroke="#ccc" strokeWidth={1} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}


