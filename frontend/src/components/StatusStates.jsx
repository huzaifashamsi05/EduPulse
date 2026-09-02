export function LoadingState({ label = "Loading..." }) {
  return (
    <div style={{ padding: "48px 0", textAlign: "center", color: "var(--ink-faint)" }}>
      <div
        aria-hidden="true"
        style={{
          width: 28,
          height: 28,
          margin: "0 auto 12px",
          border: "3px solid var(--hairline)",
          borderTopColor: "var(--accent)",
          borderRadius: "50%",
          animation: "spin 0.8s linear infinite",
        }}
      />
      <p style={{ margin: 0 }}>{label}</p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

export function ErrorState({ message, onRetry }) {
  return (
    <div
      className="card"
      style={{ borderColor: "var(--danger)", background: "var(--danger-soft)" }}
    >
      <h3 style={{ color: "var(--danger)", marginBottom: 6 }}>Something went wrong</h3>
      <p style={{ color: "var(--ink)", marginBottom: onRetry ? 14 : 0 }}>{message}</p>
      {onRetry && (
        <button className="btn btn-secondary" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title = "Nothing here yet", message }) {
  return (
    <div className="card" style={{ textAlign: "center", color: "var(--ink-soft)" }}>
      <h3 style={{ marginBottom: 6 }}>{title}</h3>
      {message && <p style={{ margin: "0 auto" }}>{message}</p>}
    </div>
  );
}

export function RiskBadge({ band }) {
  const cls =
    band === "High" ? "badge-high" : band === "Medium" ? "badge-medium" : "badge-low";
  return <span className={`badge ${cls}`}>{band} risk</span>;
}
