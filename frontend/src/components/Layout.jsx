import { NavLink, Outlet } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/students", label: "Students" },
  { to: "/predict", label: "New Prediction" },
  { to: "/insights", label: "Model Insights" },
  { to: "/about", label: "About & Responsible Use" },
];

export default function Layout() {
  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      <aside
        style={{
          width: "var(--sidebar-width)",
          flexShrink: 0,
          borderRight: "1px solid var(--hairline)",
          background: "var(--surface)",
          padding: "28px 20px",
        }}
      >
        <div style={{ marginBottom: 32 }}>
          <div style={{ fontFamily: "var(--font-display)", fontSize: "1.4rem", fontWeight: 600 }}>
            EduPulse
          </div>
          <div style={{ fontSize: "0.78rem", color: "var(--ink-faint)", marginTop: 2 }}>
            Student success signals
          </div>
        </div>

        <nav>
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              style={({ isActive }) => ({
                display: "block",
                padding: "9px 12px",
                marginBottom: 2,
                borderRadius: "var(--radius)",
                fontSize: "0.9rem",
                fontWeight: isActive ? 600 : 400,
                color: isActive ? "var(--accent-ink)" : "var(--ink-soft)",
                background: isActive ? "var(--accent-soft)" : "transparent",
                textDecoration: "none",
              })}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      <main style={{ flex: 1, padding: "36px 44px", maxWidth: 1200 }}>
        <Outlet />
      </main>
    </div>
  );
}
