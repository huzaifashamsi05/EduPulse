import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from "recharts";
import { api, ApiError } from "../api/client";
import { LoadingState, ErrorState, RiskBadge } from "../components/StatusStates";

const CLASS_COLORS = { Dropout: "#B5453D", Enrolled: "#B8802A", Graduate: "#3F7D58" };

export default function Dashboard() {
  const [summary, setSummary] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [status, setStatus] = useState("loading"); // loading | error | ready
  const [errorMsg, setErrorMsg] = useState("");

  const load = () => {
    setStatus("loading");
    Promise.all([api.analyticsSummary(), api.modelInfo()])
      .then(([s, m]) => {
        setSummary(s);
        setModelInfo(m);
        setStatus("ready");
      })
      .catch((err) => {
        setErrorMsg(err instanceof ApiError ? err.message : "Unexpected error loading the dashboard.");
        setStatus("error");
      });
  };

  useEffect(load, []);

  if (status === "loading") return <LoadingState label="Loading cohort data..." />;
  if (status === "error") return <ErrorState message={errorMsg} onRetry={load} />;

  const outcomeData = Object.entries(summary.outcome_distribution).map(([name, value]) => ({
    name,
    value,
  }));

  const courseRiskData = summary.risk_by_course.slice(0, 10).map((c) => ({
    course: `#${c.course}`,
    High: c.High || 0,
    Medium: c.Medium || 0,
    Low: c.Low || 0,
  }));

  return (
    <div>
      <h1>Cohort Dashboard</h1>
      <p style={{ color: "var(--ink-soft)" }}>
        A snapshot of predicted outcomes across the current student sample. Predictions are
        decision-support signals for human review — see{" "}
        <Link to="/about">Responsible Use</Link> before acting on them.
      </p>

      {/* --- Top stat row --- */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, margin: "24px 0" }}>
        <StatCard label="Total Students" value={summary.total_students} />
        <StatCard
          label="High Risk"
          value={summary.high_risk_count}
          sub={`${summary.high_risk_percentage}% of cohort`}
          accent="var(--risk-high)"
        />
        <StatCard
          label="Predicted Dropout"
          value={summary.outcome_distribution.Dropout || 0}
          accent="var(--risk-high)"
        />
        <StatCard
          label="Predicted Graduate"
          value={summary.outcome_distribution.Graduate || 0}
          accent="var(--risk-low)"
        />
      </div>

      {/* --- Model summary card --- */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ marginBottom: 10 }}>Model in Use</h3>
        <div style={{ display: "flex", gap: 32, flexWrap: "wrap", fontSize: "0.9rem" }}>
          <Field label="Model" value={modelInfo.model_name.replace(/_/g, " ")} />
          <Field label="Version" value={modelInfo.model_version} />
          <Field
            label="Test Macro-F1"
            value={modelInfo.test_set_metrics ? modelInfo.test_set_metrics.f1_macro.toFixed(3) : "—"}
          />
          <Field
            label="Test Accuracy"
            value={
              modelInfo.test_set_metrics
                ? `${(modelInfo.test_set_metrics.accuracy * 100).toFixed(1)}%`
                : "—"
            }
          />
        </div>
        <p style={{ marginTop: 12, fontSize: "0.85rem", color: "var(--ink-soft)" }}>
          Model selected by macro-F1, not raw accuracy, to fairly weight the minority
          "Enrolled" class. Full rationale on the <Link to="/insights">Model Insights</Link> page.
        </p>
      </div>

      {/* --- Charts --- */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.4fr", gap: 16, marginBottom: 24 }}>
        <div className="card">
          <h3>Predicted Outcome Distribution</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={outcomeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--hairline)" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {outcomeData.map((entry) => (
                  <Cell key={entry.name} fill={CLASS_COLORS[entry.name] || "var(--accent)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>Risk Band by Course (top 10 by enrollment)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={courseRiskData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--hairline)" />
              <XAxis dataKey="course" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="Low" stackId="a" fill="#3F7D58" />
              <Bar dataKey="Medium" stackId="a" fill="#B8802A" />
              <Bar dataKey="High" stackId="a" fill="#B5453D" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* --- Highest risk table --- */}
      <div className="card">
        <h3>Highest-Risk Students</h3>
        <table>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid var(--hairline)" }}>
              <Th>Student</Th>
              <Th>Course</Th>
              <Th>Predicted Outcome</Th>
              <Th>Risk</Th>
              <Th>Dropout Probability</Th>
            </tr>
          </thead>
          <tbody>
            {summary.highest_risk_students.map((s) => (
              <tr key={s.student_id} style={{ borderBottom: "1px solid var(--hairline)" }}>
                <Td>
                  <Link to={`/students/${s.student_id}`}>{s.student_id}</Link>
                </Td>
                <Td className="tabular">#{s.course}</Td>
                <Td>{s.prediction}</Td>
                <Td>
                  <RiskBadge band={s.risk_band} />
                </Td>
                <Td className="tabular">{(s.dropout_probability * 100).toFixed(1)}%</Td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StatCard({ label, value, sub, accent }) {
  return (
    <div className="card">
      <div style={{ fontSize: "0.8rem", color: "var(--ink-soft)", marginBottom: 6 }}>{label}</div>
      <div style={{ fontFamily: "var(--font-display)", fontSize: "2rem", color: accent || "var(--ink)" }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: "0.78rem", color: "var(--ink-faint)" }}>{sub}</div>}
    </div>
  );
}

function Field({ label, value }) {
  return (
    <div>
      <div style={{ fontSize: "0.75rem", color: "var(--ink-faint)" }}>{label}</div>
      <div style={{ fontWeight: 500, textTransform: "capitalize" }}>{value}</div>
    </div>
  );
}

function Th({ children }) {
  return <th style={{ padding: "8px 10px", fontSize: "0.78rem", color: "var(--ink-faint)", fontWeight: 500 }}>{children}</th>;
}
function Td({ children, className }) {
  return <td className={className} style={{ padding: "8px 10px", fontSize: "0.88rem" }}>{children}</td>;
}
