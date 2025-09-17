import React, { useState } from "react";
import { predict } from "./api";

function ResultCard({ label, probability }) {
  const color = label === "FAKE" ? "var(--accent-red)" : "var(--accent-green)";
  return (
    <div className="result-card" style={{ borderColor: color }}>
      <div className="result-header">
        <div className="result-label" style={{ backgroundColor: color }}>{label}</div>
        <div className="result-prob">{(probability * 100).toFixed(1)}%</div>
      </div>
      <div className="progress-bar">
        <div className="progress" style={{ width: `${(probability * 100).toFixed(1)}%` }} />
      </div>
      <p className="result-note">Confidence</p>
    </div>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function handleSubmit(e) {
    e?.preventDefault();
    setError("");
    setResult(null);
    if (text.trim().length < 10) {
      setError("Please enter a longer news article or paragraph (≥10 characters).");
      return;
    }
    setLoading(true);
    try {
      const res = await predict(text);
      setResult(res);
    } catch (err) {
      setError(err.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  function handlePasteSample() {
    const sample = "President announces a new initiative to boost the economy and create jobs across regions.";
    setText(sample);
  }

  return (
    <div className="page">
      <header className="topbar">
        <h1>Fake News Detector</h1>
        <p className="subtitle">NLP-based classifier — TF-IDF + Logistic Regression</p>
      </header>

      <main className="container">
        <form className="card" onSubmit={handleSubmit}>
          <label className="label">Paste a news article, paragraph, or headline</label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste text here..."
            rows={8}
          />
          <div className="controls">
            <button type="submit" className="btn primary" disabled={loading}>
              {loading ? "Analyzing…" : "Analyze"}
            </button>
            <button type="button" className="btn ghost" onClick={() => { setText(""); setResult(null); setError(""); }}>
              Clear
            </button>
            <button type="button" className="btn link" onClick={handlePasteSample}>Paste sample</button>
          </div>

          {error && <div className="error">{error}</div>}
        </form>

        <aside className="side">
          <div className="info-card card">
            <h3>How it works</h3>
            <p>We preprocess text with TF-IDF and classify it using a Logistic Regression model. This demo is intended for a resume project — not production-grade detection.</p>
            <ul>
              <li>Input: headline + article text</li>
              <li>Model: TF-IDF → LogisticRegression</li>
              <li>Output: REAL / FAKE with confidence</li>
            </ul>
          </div>

          <div className="result-area card">
            {result ? (
              <ResultCard label={result.label} probability={result.probability} />
            ) : (
              <div className="placeholder">Analysis results will appear here.</div>
            )}
          </div>

          <div className="footer card small">
            <p>Tip: For better results paste full article text (title + body).</p>
          </div>
        </aside>
      </main>

      <footer className="footer">
        <small>Made for resume/demo • Not for automated moderation</small>
      </footer>
    </div>
  );
}
