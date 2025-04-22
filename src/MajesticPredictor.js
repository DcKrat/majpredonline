// Majestic RP Predictor — с обучением на общей истории всех пользователей
import React, { useState, useEffect } from "react";
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

const ADMIN_UID = "1lIJdWqSyBhVVj3TFDEf4YMgDjY2";

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

  useEffect(() => {
    onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });
  }, []);

  useEffect(() => {
    if (!user) return;
    const loadOrCreateModel = async () => {
      try {
        const loadedModel = await tf.loadLayersModel("indexeddb://majestic-rp-model-multi");
        if (loadedModel.inputs[0].shape[1] !== depth) throw new Error("Модель несовместима");
        setModel(loadedModel);
      } catch {
        const newModel = tf.sequential();
        newModel.add(tf.layers.dense({ inputShape: [depth], units: 32, activation: "relu" }));
        newModel.add(tf.layers.dense({ units: 4, activation: "softmax" }));
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

  const trainModel = async (input, targetIndex) => {
    if (!model) return;
    const inputTensor = tf.tensor2d([input]);
    const target = new Array(4).fill(0);
    target[targetIndex] = 1;
    const outputTensor = tf.tensor2d([target]);
    await model.fit(inputTensor, outputTensor, { epochs: 3 });
    await model.save("indexeddb://majestic-rp-model-multi");
  };

  const predictForward = async () => {
    if (!model || session.length < depth) return;
    const input = session.slice(-depth).map(f => fieldToIndex[f] / 3);
    const inputTensor = tf.tensor2d([input]);
    const result = await model.predict(inputTensor).data();
    setPredictions(result);
  };

  const addResult = async (selectedField) => {
    const newSession = [...session, selectedField];
    setSession(newSession);
    if (newSession.length >= depth) {
      const input = newSession.slice(-depth).map(f => fieldToIndex[f] / 3);
      await trainModel(input, fieldToIndex[selectedField]);
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
    const allInputs = [];
    const allTargets = [];

    snapshot.forEach(doc => {
      const s = doc.data().session;
      if (!Array.isArray(s) || s.length < depth + 1) return;
      for (let i = depth; i < s.length; i++) {
        const input = s.slice(i - depth, i).map(f => fieldToIndex[f] / 3);
        const targetIndex = fieldToIndex[s[i]];
        const target = new Array(4).fill(0);
        target[targetIndex] = 1;
        allInputs.push(input);
        allTargets.push(target);
      }
    });

    if (model && allInputs.length > 0) {
      const xs = tf.tensor2d(allInputs);
      const ys = tf.tensor2d(allTargets);
      await model.fit(xs, ys, { epochs: 5 });
      await model.save("indexeddb://majestic-rp-model-multi");
      alert("Модель обучена на общей истории всех пользователей!");
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
  const predictionData = predictions ? fields.map((f, i) => ({ name: f, value: +(predictions[i] * 100).toFixed(1) })) : [];

  return (
    <div style={{ fontFamily: "sans-serif", padding: "1rem" }}>
      <h1>🎯 Majestic RP Predictor (Обновлённый)</h1>
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
