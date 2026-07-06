import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement, ArcElement,
  Title, Tooltip, Legend, PointElement, LineElement
} from "chart.js";
import { Bar, Doughnut, Line } from "react-chartjs-2";
import "./App.css";

ChartJS.register(
  CategoryScale, LinearScale, BarElement, ArcElement,
  Title, Tooltip, Legend, PointElement, LineElement
);

const API = "http://localhost:5000";

const FIELDS = [
  { key: "weight",               label: "Assessment Weight (%)",          placeholder: "e.g. 10",   min: 0,  max: 100, tip: "Percentage weight of this assessment" },
  { key: "date",                 label: "Assessment Due Day",             placeholder: "e.g. 19",   min: 1,  tip: "Day number the assessment is due" },
  { key: "assessment_type",      label: "Assessment Type (0=Exam,1=CMA,2=TMA)", placeholder: "e.g. 2", min: 0, max: 2, tip: "0=Exam, 1=CMA, 2=TMA" },
  { key: "code_module",          label: "Module Code (0–6)",              placeholder: "e.g. 0",   min: 0,  max: 6, tip: "Encoded module: 0=AAA, 1=BBB ... 6=GGG" },
  { key: "submission_delay",     label: "Submission Delay (days)",        placeholder: "e.g. 0",   tip: "Negative = submitted early, Positive = submitted late" },
  { key: "student_avg_score",    label: "Student Historical Avg Score",   placeholder: "e.g. 30 (weak) / 75 (strong)",  min: 0,  max: 100, tip: "Score <= 40 = Weak Student, Score > 40 = Strong Student" },
  { key: "assessment_difficulty",label: "Assessment Difficulty (weight)", placeholder: "e.g. 10",  min: 0,  tip: "Same as assessment weight" },
];

const initialForm = Object.fromEntries(FIELDS.map((f) => [f.key, ""]));

const chartOpts = (title) => ({
  responsive: true,
  plugins: {
    legend: { labels: { color: "#94a3b8", font: { size: 12 } } },
    title: { display: true, text: title, color: "#e2e8f0", font: { size: 13, weight: "600" }, padding: { bottom: 16 } },
    tooltip: {
      backgroundColor: "rgba(15,12,40,0.95)", borderColor: "rgba(124,58,237,0.4)", borderWidth: 1,
      titleColor: "#e2e8f0", bodyColor: "#94a3b8", padding: 10, cornerRadius: 10,
    }
  },
  scales: {
    x: { ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.04)" } },
    y: { ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.04)" } }
  }
});

const doughnutOpts = (title) => ({
  responsive: true,
  plugins: {
    legend: { labels: { color: "#94a3b8", font: { size: 12 }, padding: 16 } },
    title: { display: true, text: title, color: "#e2e8f0", font: { size: 13, weight: "600" }, padding: { bottom: 16 } },
    tooltip: {
      backgroundColor: "rgba(15,12,40,0.95)", borderColor: "rgba(124,58,237,0.4)", borderWidth: 1,
      titleColor: "#e2e8f0", bodyColor: "#94a3b8", padding: 10, cornerRadius: 10,
    }
  }
});

// ─── Topic Card ───────────────────────────────────────────────────────────────
function TopicCard({ topic, isWeak, index }) {
  const levelColor = {
    "Beginner":      "#f97316",
    "Elementary":    "#eab308",
    "Intro Coding":  "#f59e0b",
    "Concepts":      "#06b6d4",
    "Life Skill":    "#64748b",
    "Coding":        "#f59e0b",
    "Web":           "#3b82f6",
    "Tools":         "#64748b",
    "Maths":         "#06b6d4",
    "Data":          "#10b981",
    "Intermediate":  "#3b82f6",
    "AI/ML":         "#6366f1",
    "Deep Learning": "#a855f7",
    "Generative AI": "#ec4899",
    "Advanced AI":   "#8b5cf6",
    "MLOps":         "#14b8a6",
    "Research":      "#f59e0b",
    "Advanced":      "#8b5cf6",
  }[topic.level] || "#94a3b8";

  return (
    <a
      className="topic-card"
      href={topic.url}
      target="_blank"
      rel="noreferrer"
      style={{ animationDelay: `${index * 0.07}s`, textDecoration: "none" }}
    >
      <div className="topic-icon">{topic.icon}</div>
      <div className="topic-info">
        <div className="topic-title">{topic.title}</div>
        <div className="topic-footer">
          <span className="topic-level" style={{ color: levelColor, borderColor: levelColor + "40", background: levelColor + "15" }}>
            {topic.level}
          </span>
          <span className="topic-link-hint">↗ Visit</span>
        </div>
      </div>
    </a>
  );
}

// ─── Video Card ───────────────────────────────────────────────────────────────
function VideoCard({ video, index }) {
  return (
    <a className="video-card" href={video.url} target="_blank" rel="noreferrer"
      style={{ animationDelay: `${index * 0.08}s` }}>
      <div className="video-thumb-wrap">
        <img src={video.thumb} alt={video.title} className="video-thumb" />
        <div className="video-play">▶</div>
      </div>
      <div className="video-info">
        <div className="video-title">{video.title}</div>
        <div className="video-channel">📺 {video.channel}</div>
      </div>
    </a>
  );
}

// ─── Predict Page ─────────────────────────────────────────────────────────────
function PredictPage() {
  const [form, setForm]       = useState(initialForm);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");
  const [activeTab, setActiveTab] = useState("tips");

  const handleChange = (e) => { setForm({ ...form, [e.target.name]: e.target.value }); setError(""); };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const empty = FIELDS.find((f) => form[f.key] === "");
    if (empty) { setError(`Please fill in: ${empty.label}`); return; }
    setLoading(true); setResult(null); setActiveTab("tips");
    try {
      const { data } = await axios.post(`${API}/predict`, form);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.error || "Server error. Make sure the backend is running.");
    } finally { setLoading(false); }
  };

  const handleReset = () => { setForm(initialForm); setResult(null); setError(""); };

  const fillExample = (type) => {
    if (type === "weak") {
      // score <= 40 = weak
      setForm({ weight:"10", date:"19", assessment_type:"2", code_module:"0",
                submission_delay:"5", student_avg_score:"30", assessment_difficulty:"10" });
    } else {
      // score > 40 = strong
      setForm({ weight:"10", date:"19", assessment_type:"2", code_module:"0",
                submission_delay:"-1", student_avg_score:"75", assessment_difficulty:"10" });
    }
    setResult(null); setError("");
  };

  const isWeak = result?.prediction === 1;

  return (
    <div className="page">
      {/* Form */}
      <div className="card">
        <div className="card-title">📝 Student Assessment Details</div>
        <form onSubmit={handleSubmit}>
          <div className="grid">
            {FIELDS.map((field) => (
              <div className="field" key={field.key}>
                <label htmlFor={field.key}>
                  {field.label}
                  {field.tip && <span className="field-tip" title={field.tip}>ⓘ</span>}
                </label>
                <input
                  id={field.key} name={field.key} type="number"
                  placeholder={field.placeholder} value={form[field.key]}
                  onChange={handleChange} min={field.min} max={field.max}
                />
              </div>
            ))}
          </div>
          {error && <div className="error-msg">⚠️ {error}</div>}
          <div className="example-row">
            <span className="example-label">Try example:</span>
            <button type="button" className="example-btn weak-ex" onClick={() => fillExample("weak")}>⚠️ Weak Student</button>
            <button type="button" className="example-btn strong-ex" onClick={() => fillExample("strong")}>🌟 Strong Student</button>
          </div>
          <div className="btn-row">
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? <><span className="spinner" /> Analyzing...</> : "🔍 Predict Performance"}
            </button>
            <button type="button" className="btn btn-secondary" onClick={handleReset}>↺ Reset</button>
          </div>
        </form>
      </div>

      {/* Result */}
      {result && (
        <div className={`result-wrapper ${isWeak ? "weak" : "strong"}`}>

          {/* Prediction Banner */}
          <div className={`prediction-banner ${isWeak ? "weak" : "strong"}`}>
            <div className="banner-left">
              <div className="banner-icon">{isWeak ? "⚠️" : "🌟"}</div>
              <div>
                <div className="banner-label">{result.label}</div>
                <div className="banner-sublabel">{result.level_label}</div>
              </div>
            </div>
            <div className="banner-right">
              <div className="confidence-circle" style={{ "--pct": result.confidence, "--clr": isWeak ? "#ef4444" : "#10b981" }}>
                <svg viewBox="0 0 36 36">
                  <circle cx="18" cy="18" r="15.9" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="2.5" />
                  <circle cx="18" cy="18" r="15.9" fill="none"
                    stroke={isWeak ? "#ef4444" : "#10b981"} strokeWidth="2.5"
                    strokeDasharray={`${result.confidence} 100`}
                    strokeLinecap="round"
                    transform="rotate(-90 18 18)" />
                </svg>
                <span>{result.confidence}%</span>
              </div>
              <div className="conf-label">Confidence</div>
            </div>
          </div>

          {/* Inner Tabs */}
          <div className="result-tabs">
            {[
              { key: "tips",   label: "💡 Tips" },
              { key: "topics", label: "📚 Topics" },
              { key: "videos", label: "🎬 Videos" },
            ].map((t) => (
              <button key={t.key}
                className={`result-tab-btn ${activeTab === t.key ? "active" : ""} ${isWeak ? "weak" : "strong"}`}
                onClick={() => setActiveTab(t.key)}>
                {t.label}
              </button>
            ))}
          </div>

          {/* Tips */}
          {activeTab === "tips" && (
            <ul className="rec-list">
              {result.tips.map((tip, i) => (
                <li className="rec-item" key={i} style={{ animationDelay: `${i * 0.07}s` }}>
                  <span className={`rec-num ${isWeak ? "weak" : "strong"}`}>{i + 1}</span>
                  {tip}
                </li>
              ))}
            </ul>
          )}

          {/* Topics */}
          {activeTab === "topics" && (
            <div className="topics-grid">
              {result.topics.map((topic, i) => (
                <TopicCard key={i} topic={topic} isWeak={isWeak} index={i} />
              ))}
            </div>
          )}

          {/* Videos */}
          {activeTab === "videos" && (
            <div className="videos-grid">
              {result.videos.map((video, i) => (
                <VideoCard key={i} video={video} index={i} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Dashboard Page ───────────────────────────────────────────────────────────
function DashboardPage() {
  const [stats, setStats]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState("");

  useEffect(() => {
    axios.get(`${API}/stats`)
      .then(({ data }) => setStats(data))
      .catch(() => setError("Could not load stats. Make sure the backend is running."))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="page center-msg"><span className="spinner big" /></div>;
  if (error)   return <div className="page center-msg"><div className="error-msg">{error}</div></div>;

  const scoreDistChart = {
    labels: Object.keys(stats.score_distribution),
    datasets: [{
      label: "Students",
      data: Object.values(stats.score_distribution),
      backgroundColor: ["rgba(239,68,68,0.8)","rgba(249,115,22,0.8)","rgba(234,179,8,0.8)","rgba(16,185,129,0.8)","rgba(99,102,241,0.8)"],
      borderRadius: 8, borderSkipped: false,
    }]
  };

  const weakStrongChart = {
    labels: ["Strong Students", "Weak Students"],
    datasets: [{ data: [stats.strong_students, stats.weak_students], backgroundColor: ["rgba(16,185,129,0.85)","rgba(239,68,68,0.85)"], borderWidth: 0, hoverOffset: 8 }]
  };

  const moduleChart = {
    labels: Object.keys(stats.module_avg_scores),
    datasets: [{
      label: "Avg Score", data: Object.values(stats.module_avg_scores),
      borderColor: "#7c3aed", backgroundColor: "rgba(124,58,237,0.12)",
      tension: 0.45, fill: true,
      pointBackgroundColor: "#7c3aed", pointBorderColor: "#fff", pointBorderWidth: 2, pointRadius: 5,
    }]
  };

  const predChart = {
    labels: ["Strong", "Weak"],
    datasets: [{ data: [stats.predictions_strong || 0, stats.predictions_weak || 0], backgroundColor: ["rgba(16,185,129,0.85)","rgba(239,68,68,0.85)"], borderWidth: 0, hoverOffset: 8 }]
  };

  const STAT_CARDS = [
    { icon: "📊", value: stats.total_records.toLocaleString(), label: "Total Records",     cls: "" },
    { icon: "🌟", value: stats.strong_students.toLocaleString(), label: "Strong Students", cls: "strong-card" },
    { icon: "⚠️", value: stats.weak_students.toLocaleString(),   label: "Weak Students",   cls: "weak-card" },
    { icon: "🎯", value: stats.avg_score,                        label: "Average Score",   cls: "" },
    { icon: "🔍", value: stats.predictions_made,                 label: "Predictions Made",cls: "" },
  ];

  return (
    <div className="page">
      <div className="stat-grid">
        {STAT_CARDS.map((s, i) => (
          <div className={`stat-card ${s.cls}`} key={i}>
            <div className="stat-icon-wrap">{s.icon}</div>
            <div className="stat-value">{s.value}</div>
            <div className="stat-label">{s.label}</div>
          </div>
        ))}
      </div>
      <div className="chart-grid">
        <div className="card chart-card"><Bar data={scoreDistChart} options={chartOpts("Score Distribution")} /></div>
        <div className="card chart-card"><Doughnut data={weakStrongChart} options={doughnutOpts("Weak vs Strong Students")} /></div>
      </div>
      <div className="chart-grid">
        <div className="card chart-card"><Line data={moduleChart} options={chartOpts("Average Score by Module")} /></div>
        <div className="card chart-card"><Doughnut data={predChart} options={doughnutOpts("Live Predictions Breakdown")} /></div>
      </div>
    </div>
  );
}

// ─── History Page ─────────────────────────────────────────────────────────────
function HistoryPage() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchHistory = () => {
    setLoading(true);
    axios.get(`${API}/history`).then(({ data }) => setHistory(data)).finally(() => setLoading(false));
  };

  useEffect(() => { fetchHistory(); }, []);

  const clearHistory = async () => { await axios.delete(`${API}/history`); setHistory([]); };

  if (loading) return <div className="page center-msg"><span className="spinner big" /></div>;

  return (
    <div className="page">
      <div className="card">
        <div className="history-header">
          <div className="history-title">
            🕓 Prediction History
            {history.length > 0 && <span className="history-count">{history.length}</span>}
          </div>
          <div className="btn-row">
            <button className="btn btn-secondary" onClick={fetchHistory}>↺ Refresh</button>
            {history.length > 0 && <button className="btn btn-danger" onClick={clearHistory}>🗑 Clear All</button>}
          </div>
        </div>

        {history.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📭</div>
            <p>No predictions yet.<br />Go to the Predict tab to get started.</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="history-table">
              <thead>
                <tr>
                  <th>#</th><th>Timestamp</th><th>Avg Score</th>
                  <th>Delay</th><th>Weight</th><th>Result</th><th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h) => (
                  <tr key={h.id}>
                    <td style={{ color: "#64748b", fontWeight: 600 }}>{h.id}</td>
                    <td style={{ color: "#64748b", fontSize: "0.8rem" }}>{h.timestamp}</td>
                    <td><strong>{h.inputs.student_avg_score}</strong></td>
                    <td>{h.inputs.submission_delay}d</td>
                    <td>{h.inputs.weight}</td>
                    <td>
                      <span className={`badge ${h.prediction === 1 ? "badge-weak" : "badge-strong"}`}>
                        {h.prediction === 1 ? "⚠️" : "🌟"} {h.label}
                      </span>
                    </td>
                    <td>
                      <div className="mini-bar-wrap">
                        <div className="mini-bar" style={{ width: `${h.confidence}%`, background: h.prediction === 1 ? "#ef4444" : "#10b981" }} />
                      </div>
                      <span className="mini-conf">{h.confidence}%</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Root App ─────────────────────────────────────────────────────────────────
const TABS = [
  { key: "predict",   label: "🔍 Predict" },
  { key: "dashboard", label: "📊 Dashboard" },
  { key: "history",   label: "🕓 History" },
];

export default function App() {
  const [tab, setTab] = useState("predict");
  return (
    <div className="app">
      <header className="header">
        <div className="header-badge">✦ AI Powered</div>
        <h1>Learning Recommendation<br />System</h1>
        <p>Predict student performance using machine learning and get personalized learning recommendations</p>
      </header>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.key} className={`tab-btn ${tab === t.key ? "active" : ""}`} onClick={() => setTab(t.key)}>
            {t.label}
          </button>
        ))}
      </nav>
      {tab === "predict"   && <PredictPage />}
      {tab === "dashboard" && <DashboardPage />}
      {tab === "history"   && <HistoryPage />}
    </div>
  );
}
