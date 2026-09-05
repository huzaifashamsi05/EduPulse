import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { api, ApiError } from "../api/client";
import { LoadingState, ErrorState } from "../components/StatusStates";

export default function ModelInsights() {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("loading");
  const [errorMsg, setErrorMsg] = useState("");

  const load = () => {
    setStatus("loading");
    Promise.all([api.modelInfo(), api.modelInsights()])
      .then(([info, insights]) => {
        setData({ info, insights });
        setStatus("ready");
      })
      .catch((err) => {
        setErrorMsg(err instanceof ApiError ? err.message : "Could not load model insights.");
        setStatus("error");
      });
  };

  useEffect(load, []);

  if (status === "loading") return <LoadingState label="Loading model insights..." />;
  if (status === "error") return <ErrorState message={errorMsg} onRetry={load} />;

  const { insights } = data;
  const classLabels = ["Dropout", "Enrolled", "Graduate"];
  const confusionMatrix = insights.test_set_metrics.confusion_matrix;
  const perClass = insights.test_set_metrics.per_class;

  const importanceData = insights.global_feature_importance.map((f) => ({
    name: f.feature,
    importance: f.importance,
  }));

  return (
    <div>
      <h1>Model Insights</h1>
      <p style={{ color: "var(--ink-soft)" }}>
        How the final model was selected, how well it performs, and what drives its predictions.
      </p>

      {/* --- Model comparison table --- */}
      <div className="card" style={{ marginBottom: 20 }}>
        <h3>Model Comparison (5-fold cross-validation on training data)</h3>
        <table>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid var(--hairline)" }}>
              <Th>Model</Th>
              <Th>Macro-F1 (mean ± std)</Th>
              <Th>Accuracy (mean)</Th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(insights.cv_comparison).map(([name, scores]) => (
              <tr
                key={name}
                style={{
                  borderBottom: "1px solid var(--hairline)",
                  background: name === insights.selected_model ? "var(--accent-soft)" : "transparent",
                }}
              >
                <Td style={{ textTransform: "capitalize", fontWeight: name === insights.selected_model ? 600 : 400 }}>
                  {name.replace(/_/g, " ")}
                  {name === insights.selected_model && (
                    <span style={{ marginLeft: 8, fontSize: "0.72rem", color: "var(--accent-ink)" }}>
                      SELECTED
                    </span>
                  )}
                </Td>
                <Td className="tabular">
                  {scores.test_f1_macro.mean.toFixed(4)} ± {scores.test_f1_macro.std.toFixed(4)}
                </Td>
                <Td className="tabular">{(scores.test_accuracy.mean * 100).toFixed(1)}%</Td>
              </tr>
            ))}
            <tr>
              <Td style={{ color: "var(--ink-faint)" }}>Baseline (majority class)</Td>
              <Td className="tabular" style={{ color: "var(--ink-faint)" }}>
                {insights.baseline_majority_class.f1_macro.toFixed(4)}
              </Td>
              <Td className="tabular" style={{ color: "var(--ink-faint)" }}>
                {(insights.baseline_majority_class.accuracy * 100).toFixed(1)}%
              </Td>
            </tr>
          </tbody>
        </table>
        <p style={{ fontSize: "0.85rem", color: "var(--ink-soft)", marginTop: 14, marginBottom: 0 }}>
          {insights.selection_rationale}
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 20, marginBottom: 20 }}>
        {/* --- Confusion matrix --- */}
        <div className="card">
          <h3>Confusion Matrix (held-out test set)</h3>
          <table>
            <thead>
              <tr>
                <th></th>
                {classLabels.map((c) => (
                  <Th key={c} style={{ textAlign: "center" }}>
                    Predicted {c}
                  </Th>
                ))}
              </tr>
            </thead>
            <tbody>
              {confusionMatrix.map((row, i) => (
                <tr key={classLabels[i]}>
                  <Td style={{ fontWeight: 500 }}>Actual {classLabels[i]}</Td>
                  {row.map((val, j) => (
                    <Td
                      key={j}
                      className="tabular"
                      style={{
                        textAlign: "center",
                        background: i === j ? "var(--accent-soft)" : "transparent",
                        fontWeight: i === j ? 600 : 400,
                      }}
                    >
                      {val}
                    </Td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ fontSize: "0.78rem", color: "var(--ink-faint)", marginTop: 12, marginBottom: 0 }}>
            Rows = actual outcome, columns = predicted outcome. Diagonal = correct predictions.
          </p>
        </div>

        {/* --- Per-class metrics --- */}
        <div className="card">
          <h3>Per-Class Performance</h3>
          <table>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--hairline)" }}>
                <Th>Class</Th>
                <Th>Precision</Th>
                <Th>Recall</Th>
                <Th>F1</Th>
              </tr>
            </thead>
            <tbody>
              {classLabels.map((c) => (
                <tr key={c} style={{ borderBottom: "1px solid var(--hairline)" }}>
                  <Td>{c}</Td>
                  <Td className="tabular">{perClass[c].precision.toFixed(2)}</Td>
                  <Td className="tabular">{perClass[c].recall.toFixed(2)}</Td>
                  <Td className="tabular">{perClass[c]["f1-score"].toFixed(2)}</Td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ fontSize: "0.78rem", color: "var(--ink-faint)", marginTop: 12, marginBottom: 0 }}>
            "Enrolled" is the hardest class to predict — it has the fewest training examples
            and sits behaviorally between Dropout and Graduate.
          </p>
        </div>
      </div>

      {/* --- Global feature importance --- */}
      <div className="card" style={{ marginBottom: 20 }}>
        <h3>Global Feature Importance</h3>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={importanceData} layout="vertical" margin={{ left: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--hairline)" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={160} />
            <Tooltip />
            <Bar dataKey="importance" fill="var(--accent)" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <p style={{ fontSize: "0.78rem", color: "var(--ink-faint)", marginTop: 8, marginBottom: 0 }}>
          Aggregated from one-hot encoded feature contributions back to original features.
          High-cardinality categorical features (e.g. occupation codes) may appear more
          important partly due to having more encoded columns to accumulate weight across —
          worth reading as a caveat, not a precise ranking.
        </p>
      </div>

      {/* --- Metric definitions --- */}
      <div className="card">
        <h3>Metric Definitions</h3>
        <dl style={{ display: "grid", gridTemplateColumns: "160px 1fr", rowGap: 10, fontSize: "0.88rem" }}>
          <dt style={{ fontWeight: 500 }}>Accuracy</dt>
          <dd style={{ margin: 0, color: "var(--ink-soft)" }}>
            Share of all predictions that were correct. Misleading alone on imbalanced data.
          </dd>
          <dt style={{ fontWeight: 500 }}>Precision</dt>
          <dd style={{ margin: 0, color: "var(--ink-soft)" }}>
            Of students predicted into a class, the share that actually belong to it.
          </dd>
          <dt style={{ fontWeight: 500 }}>Recall</dt>
          <dd style={{ margin: 0, color: "var(--ink-soft)" }}>
            Of students who actually belong to a class, the share the model correctly caught.
          </dd>
          <dt style={{ fontWeight: 500 }}>Macro-F1</dt>
          <dd style={{ margin: 0, color: "var(--ink-soft)" }}>
            The balance of precision and recall, averaged equally across all three classes —
            used here to pick the final model because it doesn't let the majority class hide
            weak performance on the minority ones.
          </dd>
        </dl>
      </div>
    </div>
  );
}

function Th({ children, style }) {
  return (
    <th style={{ padding: "8px 10px", fontSize: "0.78rem", color: "var(--ink-faint)", fontWeight: 500, ...style }}>
      {children}
    </th>
  );
}
function Td({ children, className, style }) {
  return (
    <td className={className} style={{ padding: "8px 10px", fontSize: "0.88rem", ...style }}>
      {children}
    </td>
  );
}
