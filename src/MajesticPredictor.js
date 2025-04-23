// Majestic RP Predictor — версия с GRU и анализом невыпадений
import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { BarChart, Bar, XAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { initializeApp } from "firebase/app";
import { getFirestore, collection, addDoc, getDocs } from "firebase/firestore";
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

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const auth = getAuth(app);

const fields = ["2x", "3x", "5x", "10x"];
const fieldToIndex = { "2x": 0, "3x": 1, "5x": 2, "10x": 3 };
const barColors = { "2x": "#A0A0A0", "3x": "#E74C3C", "5x": "#2ECC71", "10x": "#FFD700" };

export default function MajesticPredictor() {
  const [user, setUser] = useState(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [session, setSession] = useState([]);
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [depth] = useState(50);
  const [totalPredictions, setTotalPredictions] = useState(0);
  const [correctPredictions, setCorrectPredictions] = useState(0);
  const trainingRef = useRef(false);
  const [missCounts, setMissCounts] = useState([0, 0, 0, 0]);

  useEffect(() => {
    onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });
  }, []);

  useEffect(() => {
    if (!user) return;
    const loadOrCreateModel = async () => {
      try {
        const loadedModel = await tf.loadLayersModel("indexeddb://majestic-rp-model-gru");
        loadedModel.compile({ optimizer: "adam", loss: "categoricalCrossentropy" });
        setModel(loadedModel);
      } catch {
        const inputLength = depth;
        const modelInput = tf.input({ shape: [inputLength, 1] });
        const gru = tf.layers.gru({ units: 32, returnSequences: false }).apply(modelInput);
        const extraInput = tf.input({ shape: [4] });
        const merged = tf.layers.concatenate().apply([gru, extraInput]);
        const output = tf.layers.dense({ units: 4, activation: "softmax" }).apply(merged);
        const newModel = tf.model({ inputs: [modelInput, extraInput], outputs: output });
        newModel.compile({ optimizer: "adam", loss: "categoricalCrossentropy" });
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

  const trainModel = async (inputSeq, extraInput, targetIndex) => {
    if (!model || trainingRef.current) return;
    trainingRef.current = true;
    try {
      const xs1 = tf.tensor3d([inputSeq.map(i => [i])]);
      const xs2 = tf.tensor2d([extraInput]);
      const target = new Array(4).fill(0);
      target[targetIndex] = 1;
      const ys = tf.tensor2d([target]);
      await model.fit([xs1, xs2], ys, { epochs: 3 });
      await model.save("indexeddb://majestic-rp-model-gru");
    } finally {
      trainingRef.current = false;
    }
  };

  const predictForward = async () => {
    if (!model || session.length < depth) return;
    const inputSeq = session.slice(-depth).map(f => fieldToIndex[f] / 3);
    const missInput = missCounts.map(c => Math.min(c / 20, 1));
    const xs1 = tf.tensor3d([inputSeq.map(i => [i])]);
    const xs2 = tf.tensor2d([missInput]);
    const result = await model.predict([xs1, xs2]).data();
    setPredictions(result);
  };

  const addResult = async (selectedField) => {
    const newSession = [...session, selectedField];
    setSession(newSession);
    const updatedMissCounts = missCounts.map((c, i) => (i === fieldToIndex[selectedField] ? 0 : c + 1));
    setMissCounts(updatedMissCounts);
    if (newSession.length >= depth) {
      const inputSeq = newSession.slice(-depth).map(f => fieldToIndex[f] / 3);
      const extraInput = updatedMissCounts.map(c => Math.min(c / 20, 1));
      await trainModel(inputSeq, extraInput, fieldToIndex[selectedField]);
      await saveToFirestore({ session: newSession });
      await predictForward();
      if (predictions) {
        const predictedMax = predictions.indexOf(Math.max(...predictions));
        if (predictedMax === fieldToIndex[selectedField]) {
          setCorrectPredictions(p => p + 1);
        }
        setTotalPredictions(p => p + 1);
      }
    }
  };

  const trainFromAllSessions = async () => {
    const snapshot = await getDocs(collection(db, "sessions"));
    const allXs1 = [], allXs2 = [], allYs = [];
    snapshot.forEach(doc => {
      const s = doc.data().session;
      if (!Array.isArray(s) || s.length < depth + 1) return;
      let miss = [0, 0, 0, 0];
      for (let i = depth; i < s.length; i++) {
        const seq = s.slice(i - depth, i).map(f => fieldToIndex[f] / 3);
        const extra = miss.map(c => Math.min(c / 20, 1));
        const target = new Array(4).fill(0);
        const idx = fieldToIndex[s[i]];
        target[idx] = 1;
        allXs1.push(seq.map(v => [v]));
        allXs2.push(extra);
        allYs.push(target);
        miss = miss.map((c, j) => (j === idx ? 0 : c + 1));
      }
    });
    if (model && allXs1.length) {
      const xs1 = tf.tensor3d(allXs1);
      const xs2 = tf.tensor2d(allXs2);
      const ys = tf.tensor2d(allYs);
      await model.fit([xs1, xs2], ys, { epochs: 5 });
      await model.save("indexeddb://majestic-rp-model-gru");
      alert("Модель GRU обучена!");
    }
  };

  const signIn = () => signInWithEmailAndPassword(auth, email, password).catch(console.error);
  const signOutUser = () => signOut(auth);

  if (!user) return (
    <div style={{ padding: 20 }}>
      <h2>🔐 Вход</h2>
      <input placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} />
      <input placeholder="Пароль" type="password" value={password} onChange={e => setPassword(e.target.value)} />
      <button onClick={signIn}>Войти</button>
    </div>
  );

  const sessionData = session.map((f, i) => ({ name: `${i + 1}`, value: 1, color: barColors[f], label: f }));
  const predictionData = predictions ? fields.map((f, i) => ({ name: f, value: +(predictions[i] * 100).toFixed(1) })) : [];

  return (
    <div style={{ fontFamily: "sans-serif", padding: "1rem" }}>
      <h1>🎯 Majestic RP Predictor (GRU версия)</h1>
      <p>👤 Пользователь: {user.email} <button onClick={signOutUser}>Выйти</button></p>
      <p>🎯 Точность предсказаний: {totalPredictions > 0 ? ((correctPredictions / totalPredictions) * 100).toFixed(1) : 0}%</p>

      <div style={{ marginBottom: 20 }}>
        {fields.map(f => (
          <button key={f} onClick={() => addResult(f)} style={{ padding: '8px 12px', backgroundColor: barColors[f], color: '#fff', border: 'none', borderRadius: 4, marginRight: 10 }}>{f}</button>
        ))}
        <button onClick={trainFromAllSessions} style={{ marginLeft: 20, padding: '8px 12px' }}>📚 Обучить по общей истории</button>
      </div>

      {predictions && (
        <div style={{ marginBottom: 20 }}>
          <h2>🔮 Прогноз:</h2>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={predictionData}>
              <XAxis dataKey="name" />
              <Tooltip />
              <Bar dataKey="value">
                {predictionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={barColors[entry.name]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <h2>📜 История сессии:</h2>
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
