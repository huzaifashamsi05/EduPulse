import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { LoadingState, ErrorState } from "../components/StatusStates";
import { PredictionCard, ExplanationCard } from "../components/PredictionResult";
import { FEATURE_LABEL_BY_KEY } from "../data/featureFields";

export default function StudentDetail() {
  const { id } = useParams();
  const [student, setStudent] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [status, setStatus] = useState("loading");
  const [errorMsg, setErrorMsg] = useState("");

  const load = () => {
    setStatus("loading");
    api
      .getStudent(id)
      .then((s) => {
        setStudent(s);
        // Prove real integration: re-run the LIVE /predict and /explain
        // endpoints with this student's stored features, rather than just
        // displaying the precomputed values (brief 4.2: "Prediction
        // endpoint is called for the selected student").
        return Promise.all([api.predict(s.features), api.explain(s.features)]);
      })
      .then(([predictionResult, explanationResult]) => {
        setPrediction(predictionResult);
        setExplanation(explanationResult);
        setStatus("ready");
      })
      .catch((err) => {
        setErrorMsg(
          err instanceof ApiError
            ? err.message
            : "Could not load this student's prediction."
        );
        setStatus("error");
      });
  };

  useEffect(load, [id]);

  if (status === "loading") return <LoadingState label="Running live prediction..." />;
  if (status === "error") return <ErrorState message={errorMsg} onRetry={load} />;

  return (
    <div>
      <Link to="/students" style={{ fontSize: "0.85rem" }}>
        ← Back to Students
      </Link>
      <h1 style={{ marginTop: 12 }}>Student {student.student_id}</h1>
      <p style={{ color: "var(--ink-soft)" }}>
        Course #{student.course} · Age {student.age_at_enrollment}
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, margin: "24px 0" }}>
        <PredictionCard prediction={prediction} />
        <ExplanationCard explanation={explanation} />
      </div>

      {/* --- Raw feature reference --- */}
      <div className="card">
        <h3>Student Record</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "10px 24px" }}>
          {Object.entries(student.features).map(([key, value]) => (
            <div key={key} style={{ fontSize: "0.83rem" }}>
              <span style={{ color: "var(--ink-faint)" }}>{FEATURE_LABEL_BY_KEY[key] || key}: </span>
              <span className="tabular">{typeof value === "number" ? value : String(value)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
