import { useState } from "react";
import { api, ApiError } from "../api/client";
import { LoadingState, ErrorState } from "../components/StatusStates";
import { PredictionCard, ExplanationCard } from "../components/PredictionResult";
import { FORM_SECTIONS, DEFAULT_FORM_VALUES } from "../data/featureFields";

export default function NewPrediction() {
  const [values, setValues] = useState(DEFAULT_FORM_VALUES);
  const [status, setStatus] = useState("idle"); // idle | loading | error | ready
  const [errorMsg, setErrorMsg] = useState("");
  const [result, setResult] = useState(null);

  const updateField = (key, raw, type) => {
    const parsed = type === "select" ? Number(raw) : raw === "" ? "" : Number(raw);
    setValues((prev) => ({ ...prev, [key]: parsed }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setStatus("loading");
    setErrorMsg("");

    Promise.all([api.predict(values), api.explain(values)])
      .then(([prediction, explanation]) => {
        setResult({ prediction, explanation });
        setStatus("ready");
      })
      .catch((err) => {
        setErrorMsg(
          err instanceof ApiError
            ? err.message
            : "Could not run the prediction. Check your inputs and try again."
        );
        setStatus("error");
      });
  };

  return (
    <div>
      <h1>New Prediction</h1>
      <p style={{ color: "var(--ink-soft)" }}>
        Enter a student's academic and demographic record to get a live model prediction.
        Every field below is sent directly to the prediction API — nothing here is
        precomputed or hardcoded.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: result ? "1fr 1fr" : "1fr", gap: 24, alignItems: "start" }}>
        <form onSubmit={handleSubmit}>
          {FORM_SECTIONS.map((section) => (
            <div key={section.title} className="card" style={{ marginBottom: 16 }}>
              <h3>{section.title}</h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px 16px" }}>
                {section.fields.map((field) => (
                  <div key={field.key}>
                    <label htmlFor={field.key}>{field.label}</label>
                    {field.type === "select" ? (
                      <select
                        id={field.key}
                        value={values[field.key]}
                        onChange={(e) => updateField(field.key, e.target.value, "select")}
                      >
                        {field.options.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        id={field.key}
                        type="number"
                        step={field.step ?? 1}
                        min={field.min}
                        max={field.max}
                        value={values[field.key]}
                        onChange={(e) => updateField(field.key, e.target.value, "number")}
                        required
                      />
                    )}
                    {field.help && (
                      <div style={{ fontSize: "0.72rem", color: "var(--ink-faint)", marginTop: 3 }}>
                        {field.help}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}

          <button type="submit" className="btn btn-primary" disabled={status === "loading"}>
            {status === "loading" ? "Running prediction..." : "Run Prediction"}
          </button>
        </form>

        {/* --- Results panel --- */}
        <div style={{ position: "sticky", top: 20 }}>
          {status === "idle" && !result && (
            <div className="card" style={{ color: "var(--ink-faint)", textAlign: "center" }}>
              Fill out the form and click "Run Prediction" to see results here.
            </div>
          )}
          {status === "loading" && <LoadingState label="Calling the prediction API..." />}
          {status === "error" && (
            <ErrorState message={errorMsg} onRetry={() => setStatus("idle")} />
          )}
          {status === "ready" && result && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <PredictionCard prediction={result.prediction} />
              <ExplanationCard explanation={result.explanation} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
