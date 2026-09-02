import { RiskBadge } from "./StatusStates";

const CLASS_COLORS = { Dropout: "#B5453D", Enrolled: "#B8802A", Graduate: "#3F7D58" };

export function PredictionCard({ prediction }) {
  return (
    <div className="card">
      <h3>Predicted Outcome</h3>
      <div
        style={{
          fontFamily: "var(--font-display)",
          fontSize: "1.7rem",
          color: CLASS_COLORS[prediction.prediction],
          marginBottom: 4,
        }}
      >
        {prediction.prediction}
      </div>
      <RiskBadge band={prediction.risk_band} />

      <div style={{ marginTop: 18 }}>
        {Object.entries(prediction.probabilities).map(([label, value]) => (
          <div key={label} style={{ marginBottom: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem" }}>
              <span>{label}</span>
              <span className="tabular">{(value * 100).toFixed(1)}%</span>
            </div>
            <div style={{ background: "var(--surface-sunken)", borderRadius: 4, height: 6 }}>
              <div
                style={{
                  width: `${value * 100}%`,
                  background: CLASS_COLORS[label],
                  height: 6,
                  borderRadius: 4,
                }}
              />
            </div>
          </div>
        ))}
      </div>
      <p style={{ fontSize: "0.78rem", color: "var(--ink-faint)", marginTop: 14, marginBottom: 0 }}>
        Model version {prediction.model_version} · live inference via POST /predict
      </p>
    </div>
  );
}

export function ExplanationCard({ explanation }) {
  return (
    <div className="card">
      <h3>Why This Prediction?</h3>
      <p style={{ fontSize: "0.85rem", color: "var(--ink-soft)" }}>
        The factors below moved the model's output toward or away from the predicted class.
      </p>
      {explanation.top_factors.map((f) => (
        <div key={f.feature} style={{ marginBottom: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem" }}>
            <span>{f.feature}</span>
            <span
              className="tabular"
              style={{ color: f.direction.startsWith("toward") ? "var(--accent-ink)" : "var(--ink-faint)" }}
            >
              {f.direction.startsWith("toward") ? "↑" : "↓"} {f.impact}
            </span>
          </div>
        </div>
      ))}
      <p style={{ fontSize: "0.78rem", color: "var(--ink-faint)", marginTop: 14, marginBottom: 0 }}>
        {explanation.note}
      </p>
    </div>
  );
}
