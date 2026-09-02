import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { LoadingState, ErrorState, EmptyState, RiskBadge } from "../components/StatusStates";

export default function Students() {
  const [students, setStudents] = useState([]);
  const [status, setStatus] = useState("loading");
  const [errorMsg, setErrorMsg] = useState("");
  const [riskFilter, setRiskFilter] = useState("");
  const [search, setSearch] = useState("");

  const load = () => {
    setStatus("loading");
    api
      .listStudents({ riskBand: riskFilter || undefined, limit: 300 })
      .then((data) => {
        setStudents(data);
        setStatus("ready");
      })
      .catch((err) => {
        setErrorMsg(err instanceof ApiError ? err.message : "Could not load students.");
        setStatus("error");
      });
  };

  useEffect(load, [riskFilter]);

  const filtered = students.filter((s) => {
    if (!search.trim()) return true;
    const q = search.trim().toLowerCase();
    return s.student_id.toLowerCase().includes(q) || String(s.course).includes(q);
  });

  return (
    <div>
      <h1>Students</h1>
      <p style={{ color: "var(--ink-soft)" }}>
        Sample cohort with model-predicted outcomes. Search by student ID or course code.
      </p>

      <div style={{ display: "flex", gap: 12, margin: "20px 0", maxWidth: 480 }}>
        <input
          placeholder="Search student ID or course..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select value={riskFilter} onChange={(e) => setRiskFilter(e.target.value)}>
          <option value="">All risk levels</option>
          <option value="High">High risk</option>
          <option value="Medium">Medium risk</option>
          <option value="Low">Low risk</option>
        </select>
      </div>

      {status === "loading" && <LoadingState label="Loading students..." />}
      {status === "error" && <ErrorState message={errorMsg} onRetry={load} />}

      {status === "ready" && filtered.length === 0 && (
        <EmptyState title="No students match" message="Try a different search term or filter." />
      )}

      {status === "ready" && filtered.length > 0 && (
        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <table>
            <thead>
              <tr style={{ textAlign: "left", borderBottom: "1px solid var(--hairline)", background: "var(--surface-sunken)" }}>
                <Th>Student ID</Th>
                <Th>Course</Th>
                <Th>Age</Th>
                <Th>Predicted Outcome</Th>
                <Th>Risk</Th>
                <Th>Confidence</Th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((s) => {
                const topProb = Math.max(...Object.values(s.probabilities));
                return (
                  <tr key={s.student_id} style={{ borderBottom: "1px solid var(--hairline)" }}>
                    <Td>
                      <Link to={`/students/${s.student_id}`} style={{ fontWeight: 500 }}>
                        {s.student_id}
                      </Link>
                    </Td>
                    <Td className="tabular">#{s.course}</Td>
                    <Td>{s.age_at_enrollment}</Td>
                    <Td>{s.prediction}</Td>
                    <Td>
                      <RiskBadge band={s.risk_band} />
                    </Td>
                    <Td className="tabular">{(topProb * 100).toFixed(0)}%</Td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {status === "ready" && (
        <p style={{ marginTop: 12, fontSize: "0.8rem", color: "var(--ink-faint)" }}>
          Showing {filtered.length} of {students.length} students.
        </p>
      )}
    </div>
  );
}

function Th({ children }) {
  return <th style={{ padding: "10px 14px", fontSize: "0.78rem", color: "var(--ink-faint)", fontWeight: 500 }}>{children}</th>;
}
function Td({ children, className }) {
  return <td className={className} style={{ padding: "10px 14px", fontSize: "0.88rem" }}>{children}</td>;
}
