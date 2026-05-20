import { useState } from "react";

const API_URL = "https://credit-card-fraud-model.onrender.com";

const FEATURES = [
  { key: "V14",           label: "Authentication Pattern",      help: "Lower = unusual auth patterns",       min: -20, max: 10,    step: 0.01, default: 0,  isAmount: false },
  { key: "V17",           label: "Spending Velocity",           help: "Sudden spikes = fraud signal",        min: -20, max: 10,    step: 0.01, default: 0,  isAmount: false },
  { key: "V12",           label: "Merchant Category Deviation", help: "Deviation from typical merchants",    min: -20, max: 10,    step: 0.01, default: 0,  isAmount: false },
  { key: "V10",           label: "Geographic Anomaly Score",    help: "Card used in distant locations",      min: -20, max: 10,    step: 0.01, default: 0,  isAmount: false },
  { key: "V4",            label: "Card Usage Frequency",        help: "Unusually high frequency = suspect",  min: -10, max: 20,    step: 0.01, default: 0,  isAmount: false },
  { key: "V11",           label: "Time-of-Day Behaviour",       help: "Matches normal activity hours?",      min: -10, max: 20,    step: 0.01, default: 0,  isAmount: false },
  { key: "Amount_scaled", label: "Transaction Amount (€)",      help: "Actual euro value",                   min: 0,   max: 25000, step: 1,    default: 50, isAmount: true  },
];

const RISK_CONFIG = {
  low:      { color: "#22c55e", bg: "#f0fdf4", border: "#86efac", label: "Low Risk",      desc: "Transaction appears normal.",          icon: "●" },
  moderate: { color: "#f59e0b", bg: "#fffbeb", border: "#fde68a", label: "Moderate Risk", desc: "Some unusual signals detected.",        icon: "◆" },
  high:     { color: "#f97316", bg: "#fff7ed", border: "#fdba74", label: "High Risk",      desc: "Flag for manual review.",              icon: "▲" },
  critical: { color: "#ef4444", bg: "#fef2f2", border: "#fca5a5", label: "Critical Risk",  desc: "Block this transaction immediately.",  icon: "⬟" },
};

export default function App() {
  const initValues = Object.fromEntries(FEATURES.map(f => [f.key, f.default]));
  const [values,  setValues]  = useState(initValues);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);
  const [activeTab, setActiveTab] = useState("waterfall");

  const handleChange = (key, val) =>
    setValues(prev => ({ ...prev, [key]: parseFloat(val) }));

  const handleAnalyse = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(values),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
      setActiveTab("waterfall");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setValues(initValues);
    setResult(null);
    setError(null);
  };

  const risk = result ? RISK_CONFIG[result.risk_level] : null;
  const maxAbs = result
    ? Math.max(...result.contributions.map(c => Math.abs(c.shap_value)))
    : 1;

  return (
    <div style={{ minHeight: "100vh", background: "#0f0f13", fontFamily: "'IBM Plex Mono', 'Courier New', monospace", color: "#e2e8f0" }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{ borderBottom: "1px solid #1e2030", padding: "1.25rem 2rem", display: "flex", alignItems: "center", gap: "1rem", background: "#0f0f13" }}>
        <div style={{ width: 36, height: 36, border: "1.5px solid #7c3aed", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 }}>🛡️</div>
        <div>
          <div style={{ fontFamily: "'IBM Plex Mono'", fontWeight: 600, fontSize: 15, color: "#f1f5f9", letterSpacing: "0.05em" }}>FRAUD_DETECTOR</div>
          <div style={{ fontSize: 11, color: "#64748b", letterSpacing: "0.08em" }}>XGBoost · SHAP · PR-AUC 0.78</div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#22c55e" }}></div>
          <span style={{ fontSize: 12, color: "#64748b" }}>API CONNECTED</span>
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "2rem" }}>

        {/* Input Panel */}
        <div style={{ background: "#13131a", border: "1px solid #1e2030", borderRadius: 12, padding: "1.75rem", marginBottom: "1.5rem" }}>
          <div style={{ fontFamily: "'IBM Plex Mono'", fontSize: 11, letterSpacing: "0.12em", color: "#7c3aed", marginBottom: "1.25rem", textTransform: "uppercase" }}>
            // transaction_inputs
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1.25rem" }}>
            {FEATURES.map(feat => (
              <div key={feat.key}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                  <label style={{ fontSize: 12, color: "#94a3b8", fontFamily: "'IBM Plex Mono'" }}>{feat.label}</label>
                  <span style={{ fontSize: 12, fontWeight: 600, color: "#7c3aed", fontFamily: "'IBM Plex Mono'" }}>
                    {feat.isAmount ? `€${values[feat.key].toFixed(0)}` : values[feat.key].toFixed(2)}
                  </span>
                </div>
                {feat.isAmount ? (
                  <input
                    type="number"
                    min={feat.min} max={feat.max} step={feat.step}
                    value={values[feat.key]}
                    onChange={e => handleChange(feat.key, e.target.value)}
                    style={{ width: "100%", background: "#0f0f13", border: "1px solid #2d2d3d", borderRadius: 6, padding: "8px 10px", color: "#e2e8f0", fontFamily: "'IBM Plex Mono'", fontSize: 13, boxSizing: "border-box" }}
                  />
                ) : (
                  <>
                    <input
                      type="range"
                      min={feat.min} max={feat.max} step={feat.step}
                      value={values[feat.key]}
                      onChange={e => handleChange(feat.key, e.target.value)}
                      style={{ width: "100%", accentColor: "#7c3aed", cursor: "pointer" }}
                    />
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#475569", marginTop: 2 }}>
                      <span>{feat.min}</span><span>{feat.max}</span>
                    </div>
                  </>
                )}
                <div style={{ fontSize: 10, color: "#475569", marginTop: 4 }}>{feat.help}</div>
              </div>
            ))}
          </div>

          <div style={{ display: "flex", gap: 10, marginTop: "1.75rem" }}>
            <button
              onClick={handleAnalyse}
              disabled={loading}
              style={{ flex: 1, background: loading ? "#2d2d3d" : "#7c3aed", border: "none", borderRadius: 8, padding: "12px 0", color: "#fff", fontFamily: "'IBM Plex Mono'", fontWeight: 600, fontSize: 13, cursor: loading ? "default" : "pointer", letterSpacing: "0.05em", transition: "background 0.2s" }}
            >
              {loading ? "ANALYSING..." : "▶ ANALYSE_TRANSACTION"}
            </button>
            <button
              onClick={handleReset}
              style={{ background: "transparent", border: "1px solid #2d2d3d", borderRadius: 8, padding: "12px 20px", color: "#64748b", fontFamily: "'IBM Plex Mono'", fontSize: 13, cursor: "pointer" }}
            >
              RESET
            </button>
          </div>
        </div>

        {error && (
          <div style={{ background: "#1a0a0a", border: "1px solid #7f1d1d", borderRadius: 10, padding: "1rem 1.25rem", marginBottom: "1.5rem", color: "#fca5a5", fontSize: 13, fontFamily: "'IBM Plex Mono'" }}>
            ⚠ ERROR: {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <>
            {/* Verdict row */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "1rem", marginBottom: "1.5rem" }}>

              {/* Verdict card */}
              <div style={{ background: result.is_fraud ? "#1a0a0a" : "#0a1a0f", border: `1px solid ${result.is_fraud ? "#7f1d1d" : "#14532d"}`, borderRadius: 12, padding: "1.5rem", textAlign: "center" }}>
                <div style={{ fontSize: 11, letterSpacing: "0.12em", color: result.is_fraud ? "#ef4444" : "#22c55e", marginBottom: 10, fontFamily: "'IBM Plex Mono'" }}>
                  {result.is_fraud ? "// FRAUD_DETECTED" : "// TRANSACTION_CLEAR"}
                </div>
                <div style={{ fontSize: 42, fontWeight: 700, color: result.is_fraud ? "#ef4444" : "#22c55e", fontFamily: "'IBM Plex Mono'", lineHeight: 1 }}>
                  {(result.fraud_probability * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: 12, color: "#64748b", marginTop: 6 }}>fraud probability</div>
              </div>

              {/* Risk level */}
              <div style={{ background: "#13131a", border: "1px solid #1e2030", borderRadius: 12, padding: "1.5rem" }}>
                <div style={{ fontSize: 11, letterSpacing: "0.12em", color: "#7c3aed", marginBottom: 12, fontFamily: "'IBM Plex Mono'" }}>// risk_level</div>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                  <span style={{ fontSize: 20, color: risk.color }}>{risk.icon}</span>
                  <span style={{ fontSize: 15, fontWeight: 600, color: risk.color, fontFamily: "'IBM Plex Mono'" }}>{risk.label}</span>
                </div>
                <div style={{ fontSize: 12, color: "#94a3b8" }}>{risk.desc}</div>
                <div style={{ marginTop: 14, height: 6, background: "#1e2030", borderRadius: 3, overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${result.fraud_probability * 100}%`, background: risk.color, borderRadius: 3, transition: "width 0.8s ease" }}></div>
                </div>
              </div>

              {/* Stats */}
              <div style={{ background: "#13131a", border: "1px solid #1e2030", borderRadius: 12, padding: "1.5rem" }}>
                <div style={{ fontSize: 11, letterSpacing: "0.12em", color: "#7c3aed", marginBottom: 12, fontFamily: "'IBM Plex Mono'" }}>// model_output</div>
                {[
                  ["P(fraud)",    `${(result.fraud_probability * 100).toFixed(4)}%`],
                  ["P(legit)",    `${((1 - result.fraud_probability) * 100).toFixed(4)}%`],
                  ["Base value",  result.shap_base_value.toFixed(4)],
                  ["Decision",    result.is_fraud ? "BLOCK" : "ALLOW"],
                ].map(([k, v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #1e2030", fontSize: 12 }}>
                    <span style={{ color: "#64748b" }}>{k}</span>
                    <span style={{ color: k === "Decision" ? (result.is_fraud ? "#ef4444" : "#22c55e") : "#e2e8f0", fontFamily: "'IBM Plex Mono'", fontWeight: 500 }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* SHAP section */}
            <div style={{ background: "#13131a", border: "1px solid #1e2030", borderRadius: 12, padding: "1.75rem" }}>
              <div style={{ fontFamily: "'IBM Plex Mono'", fontSize: 11, letterSpacing: "0.12em", color: "#7c3aed", marginBottom: "1.25rem" }}>
                // shap_explanation
              </div>

              {/* Tabs */}
              <div style={{ display: "flex", gap: 8, marginBottom: "1.5rem" }}>
                {["waterfall", "table"].map(tab => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    style={{ background: activeTab === tab ? "#7c3aed" : "transparent", border: `1px solid ${activeTab === tab ? "#7c3aed" : "#2d2d3d"}`, borderRadius: 6, padding: "6px 14px", color: activeTab === tab ? "#fff" : "#64748b", fontFamily: "'IBM Plex Mono'", fontSize: 11, cursor: "pointer", letterSpacing: "0.06em" }}
                  >
                    {tab === "waterfall" ? "WATERFALL" : "TABLE"}
                  </button>
                ))}
              </div>

              {activeTab === "waterfall" && (
                <div>
                  <div style={{ fontSize: 11, color: "#475569", marginBottom: "1rem", fontFamily: "'IBM Plex Mono'" }}>
                    base_value: {result.shap_base_value.toFixed(4)} → output: {result.fraud_probability.toFixed(4)}
                  </div>
                  {result.contributions.map((c, i) => {
                    const pct = (Math.abs(c.shap_value) / maxAbs) * 100;
                    const isFraud = c.shap_value > 0;
                    const barColor = isFraud ? "#ef4444" : "#3b82f6";
                    return (
                      <div key={c.feature} style={{ display: "grid", gridTemplateColumns: "180px 1fr 80px", alignItems: "center", gap: 12, marginBottom: 10 }}>
                        <div style={{ fontSize: 11, color: "#94a3b8", fontFamily: "'IBM Plex Mono'", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.label}</div>
                        <div style={{ position: "relative", height: 22, background: "#0f0f13", borderRadius: 4, overflow: "hidden" }}>
                          <div style={{ position: "absolute", left: isFraud ? "50%" : `${50 - pct / 2}%`, width: `${pct / 2}%`, height: "100%", background: barColor, opacity: 0.85, borderRadius: 2 }}></div>
                          <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "#2d2d3d" }}></div>
                        </div>
                        <div style={{ fontSize: 11, color: isFraud ? "#ef4444" : "#3b82f6", textAlign: "right", fontFamily: "'IBM Plex Mono'" }}>
                          {c.shap_value > 0 ? "+" : ""}{c.shap_value.toFixed(4)}
                        </div>
                      </div>
                    );
                  })}
                  <div style={{ display: "flex", gap: 16, marginTop: "1rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "#64748b" }}>
                      <div style={{ width: 12, height: 12, background: "#ef4444", borderRadius: 2 }}></div>toward fraud
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "#64748b" }}>
                      <div style={{ width: 12, height: 12, background: "#3b82f6", borderRadius: 2 }}></div>toward legit
                    </div>
                  </div>
                </div>
              )}

              {activeTab === "table" && (
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, fontFamily: "'IBM Plex Mono'" }}>
                    <thead>
                      <tr style={{ borderBottom: "1px solid #2d2d3d" }}>
                        {["Feature", "Input", "SHAP Value", "Effect"].map(h => (
                          <th key={h} style={{ textAlign: "left", padding: "8px 12px", color: "#475569", fontWeight: 500, fontSize: 11, letterSpacing: "0.08em" }}>{h.toUpperCase()}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.contributions.map(c => (
                        <tr key={c.feature} style={{ borderBottom: "1px solid #1e2030" }}>
                          <td style={{ padding: "8px 12px", color: "#94a3b8" }}>{c.label}</td>
                          <td style={{ padding: "8px 12px", color: "#e2e8f0" }}>
                            {c.feature === "Amount_scaled" ? `€${c.input_value.toFixed(2)}` : c.input_value.toFixed(4)}
                          </td>
                          <td style={{ padding: "8px 12px", color: c.shap_value > 0 ? "#ef4444" : "#3b82f6", fontWeight: 600 }}>
                            {c.shap_value > 0 ? "+" : ""}{c.shap_value.toFixed(6)}
                          </td>
                          <td style={{ padding: "8px 12px" }}>
                            <span style={{ background: c.effect === "fraud" ? "#450a0a" : "#0c1445", color: c.effect === "fraud" ? "#f87171" : "#60a5fa", padding: "3px 8px", borderRadius: 4, fontSize: 10 }}>
                              {c.effect === "fraud" ? "↑ FRAUD" : "↓ LEGIT"}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}

        {/* Empty state */}
        {!result && !loading && !error && (
          <div style={{ textAlign: "center", padding: "3rem 1rem", color: "#334155" }}>
            <div style={{ fontSize: 48, marginBottom: "1rem" }}>◈</div>
            <div style={{ fontFamily: "'IBM Plex Mono'", fontSize: 13, color: "#475569" }}>
              Adjust inputs and click ANALYSE_TRANSACTION
            </div>
            <div style={{ fontFamily: "'IBM Plex Mono'", fontSize: 11, color: "#334155", marginTop: 8 }}>
              Returns fraud probability + SHAP feature contributions
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{ marginTop: "2rem", borderTop: "1px solid #1e2030", paddingTop: "1rem", display: "flex", gap: "1.5rem", fontSize: 10, color: "#334155", fontFamily: "'IBM Plex Mono'", flexWrap: "wrap" }}>
          <span>MODEL: XGBoost</span>
          <span>FEATURES: V14 V17 V12 V10 V4 V11 Amount_scaled</span>
          <span>EXPLAINER: SHAP TreeExplainer</span>
          <span>DATASET: Kaggle ULB (284,807 txns)</span>
        </div>
      </div>
    </div>
  );
}