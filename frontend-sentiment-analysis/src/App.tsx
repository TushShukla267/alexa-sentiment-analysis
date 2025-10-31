// frontend/src/App.tsx
import "./App.css";
import { useState } from "react";
import axios from "axios";

type PredictionResp = {
  prediction: number | string;
  probability?: number | null;
  model?: string;
  probs?: { "0": number; "1": number } | { 0: number; 1: number } | Record<string, number>;
};

type FilePrediction = PredictionResp & { text: string };

const makeBarStyle = (p: number) => ({ width: `${Math.round(p * 100)}%` });

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:5000/predict";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [model, setModel] = useState("xgb");
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResp | null>(null);
  const [filePredictions, setFilePredictions] = useState<FilePrediction[] | null>(null);
  const [history, setHistory] = useState<(PredictionResp & { text: string })[]>([]);
  const [error, setError] = useState<string | null>(null);

  const toHistoryEntry = (p: PredictionResp, t = "") => ({ ...p, text: t });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    setPrediction(null);
    setFilePredictions(null);

    try {
      const formData = new FormData();
      if (file) formData.append("file", file);
      if (text) formData.append("text", text);
      formData.append("model", model);

      const res = await axios.post(BACKEND_URL, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 30000,
      });

      // Backend returns:
      // - single-text: { prediction, probability, probs, model }
      // - file: { model: 'xgb', predictions: [{text, prediction, probability, probs}, ...] }
      const data = res.data;

      // If file upload -> array of predictions
      if (data && Array.isArray(data.predictions)) {
        const preds: FilePrediction[] = data.predictions.map((p: any) => ({
          text: p.text ?? "",
          prediction: typeof p.prediction === "string" ? p.prediction : Number(p.prediction),
          probability: p.probability ?? null,
          model: data.model ?? p.model,
          probs: p.probs ?? undefined,
        }));

        setFilePredictions(preds);

        // add latest CSV predictions to history (prepend)
        setHistory(prev => {
          const items = preds.slice(0, 10).map(p => toHistoryEntry({ prediction: p.prediction, probability: p.probability, model: p.model }, p.text));
          return [...items, ...prev].slice(0, 50); // keep some history
        });

        // also show the first row as the "current" single prediction for pill UI
        if (preds.length > 0) {
          setPrediction({
            prediction: preds[0].prediction,
            probability: preds[0].probability ?? undefined,
            model: preds[0].model,
            probs: preds[0].probs,
          });
        }
      } else {
        // single text response
        const single: PredictionResp = {
          prediction: typeof data.prediction === "string" ? data.prediction : Number(data.prediction),
          probability: data.probability ?? undefined,
          model: data.model ?? model,
          probs: data.probs ?? undefined,
        };
        setPrediction(single);
        setHistory(prev => [toHistoryEntry(single, text), ...prev].slice(0, 10));
      }
    } catch (err: any) {
      console.error(err);
      if (err?.response?.data?.error) setError(String(err.response.data.error));
      else setError("Error: Unable to get prediction. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const prettyClass = (pred: any) => {
    if (typeof pred === "string")
      return pred.toLowerCase().includes("pos")
        ? "text-green-700 bg-green-50 ring-green-200"
        : "text-red-700 bg-red-50 ring-red-200";
    return pred === 1 || pred === "1" || pred === "Positive"
      ? "text-green-700 bg-green-50 ring-green-200"
      : "text-red-700 bg-red-50 ring-red-200";
  };

  const downloadCSV = () => {
    if (!filePredictions || filePredictions.length === 0) return;
    const header = ["text", "prediction", "probability"];
    const rows = filePredictions.map(p => [
      `"${(p.text || "").replace(/"/g, '""')}"`,
      String(p.prediction),
      p.probability != null ? (p.probability * 100).toFixed(2) : "",
    ]);
    const csv = [header.join(","), ...rows.map(r => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `predictions_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <section className="hero">
        <h1>Understand the emotions behind the words. 😊</h1>
        <p>
          Text sentiment prediction helps us understand emotions, detect opinions,
          and predict product feedback. This system analyzes Amazon Alexa reviews
          to determine whether they express positive or negative sentiments.
        </p>
      </section>

      {/* Form Section */}
      <section className="form-section">
        <h2>Text Sentiment Prediction</h2>
        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <label htmlFor="csv">Upload your CSV file</label>
            <input
              type="file"
              id="csv"
              accept=".csv"
              onChange={(e) => {
                setFile(e.target.files?.[0] || null);
                // clear previous file predictions when user picks a new file
                setFilePredictions(null);
              }}
            />
          </div>

          <div className="input-group">
            <label htmlFor="text">Enter Text for Prediction</label>
            <textarea
              id="text"
              rows={3}
              placeholder="Example: I love my new Alexa!"
              value={text}
              onChange={(e) => setText(e.target.value)}
            ></textarea>
          </div>

          <div className="flex items-center gap-4 mb-6">
            <label className="text-gray-700 font-medium">Select Model:</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="border border-gray-300 rounded-lg p-2 focus:ring-2 focus:ring-blue-500"
            >
              <option value="xgb">XGBoost</option>
              <option value="rf">Random Forest</option>
              <option value="dt">Decision Tree</option>
            </select>
            <button
              type="submit"
              disabled={loading}
              className={`ml-auto bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition duration-200 ${
                loading ? "opacity-70" : ""
              }`}
            >
              {loading ? "Predicting..." : "Predict"}
            </button>
          </div>
        </form>

        {error && <div className="text-red-600 mb-4">{error}</div>}

        {/* Prediction Result */}
        {prediction && (
          <div className="prediction-box">
            <h3>Prediction Result</h3>
            <div
              className={`inline-flex items-center gap-3 px-3 py-1 rounded-full text-sm font-medium ${prettyClass(
                prediction.prediction
              )} ring-1`}
            >
              Sentiment:{" "}
              {typeof prediction.prediction === "string"
                ? prediction.prediction
                : prediction.prediction === 1
                ? "Positive"
                : "Negative"}
            </div>

            {prediction.probability != null && (
              <div className="mt-3">
                <div className="text-xs text-gray-500">Probability</div>
                <div className="mt-1 bg-gray-100 rounded-full h-4 overflow-hidden">
                  <div
                    className="h-4 rounded-full bg-indigo-600"
                    style={makeBarStyle(prediction.probability as number)}
                  />
                </div>
                <div className="text-sm mt-2">
                  {(prediction.probability! * 100).toFixed(1)}% positive
                </div>
              </div>
            )}
          </div>
        )}

        {/* Graph Placeholder */}
        <div className="graph-box">
          <h3>Graph Result</h3>
          <p>Coming soon...</p>
        </div>
      </section>

      {/* CSV Results (if any) */}
      {filePredictions && (
        <section className="max-w-xl mx-auto mb-6">
          <div className="rounded-lg border p-3 bg-gray-50">
            <div className="flex justify-between items-center">
              <div className="text-xs text-gray-500">CSV Predictions ({filePredictions.length})</div>
              <div className="space-x-2">
                <button
                  onClick={downloadCSV}
                  className="bg-indigo-600 text-white text-xs px-3 py-1 rounded hover:opacity-90"
                >
                  Download CSV
                </button>
              </div>
            </div>

            <div className="mt-2 space-y-2 max-h-72 overflow-auto">
              {filePredictions.map((p, i) => (
                <div key={i} className="p-2 rounded bg-white shadow-sm border text-xs">
                  <div className="flex items-center justify-between">
                    <div className="font-medium">{p.model?.toUpperCase() || model}</div>
                    <div
                      className={`text-xs px-2 py-0.5 rounded ${
                        (p.prediction === 1 || p.prediction === "Positive")
                          ? "bg-green-100 text-green-800"
                          : "bg-red-100 text-red-800"
                      }`}
                    >
                      {(p.prediction === 1 || p.prediction === "Positive") ? "Positive" : "Negative"}
                    </div>
                  </div>

                  <div className="mt-1 text-xs text-gray-600 truncate">{p.text}</div>

                  {p.probability != null && (
                    <div className="mt-2 text-xs text-gray-500">
                      Prob: {(p.probability * 100).toFixed(1)}%
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* Prediction History */}
      <section className="max-w-xl mx-auto mb-12">
        <div className="rounded-lg border p-3 bg-gray-50">
          <div className="text-xs text-gray-500">History (last 10)</div>
          {history.length === 0 ? (
            <div className="text-sm text-gray-400 mt-2">No previous predictions yet.</div>
          ) : (
            <div className="mt-2 space-y-2 max-h-72 overflow-auto">
              {history.slice(0, 10).map((h, i) => (
                <div key={i} className="p-2 rounded bg-white shadow-sm border">
                  <div className="flex items-center justify-between text-sm">
                    <div className="font-medium">{h.model?.toUpperCase()}</div>
                    <div
                      className={`text-xs px-2 py-0.5 rounded ${
                        h.prediction === 1 || h.prediction === "Positive"
                          ? "bg-green-100 text-green-800"
                          : "bg-red-100 text-red-800"
                      }`}
                    >
                      {h.prediction === 1 || h.prediction === "Positive" ? "Positive" : "Negative"}
                    </div>
                  </div>
                  <div className="mt-1 text-xs text-gray-600 truncate">{h.text}</div>
                  {h.probability != null && (
                    <div className="mt-2 text-xs text-gray-500">Prob: {(h.probability * 100).toFixed(1)}%</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      <footer className="text-sm text-gray-500 text-center pb-6">
        Made for NLP Mini Project • Connects to Flask API at /predict
      </footer>
    </div>
  );
}
