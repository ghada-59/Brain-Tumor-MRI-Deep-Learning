import { useEffect, useState } from "react";
import { BrowserRouter, NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Brain, ChartColumn, Layers, Moon, ScanSearch, ShieldAlert, ShieldCheck, Sun, TriangleAlert, Home, Repeat } from "lucide-react";
import Overview from "./pages/Overview.jsx";
import Methodology from "./pages/Methodology.jsx";
import Evaluation from "./pages/Evaluation.jsx";
import Explainability from "./pages/Explainability.jsx";
import Inference from "./pages/Inference.jsx";
import Limitations from "./pages/Limitations.jsx";
import Reproducibility from "./pages/Reproducibility.jsx";

const NAV = [
  { to: "/", label: "Overview", icon: Home, title: "Overview", end: true },
  { to: "/methodology", label: "Data & methodology", icon: Layers, title: "Data & methodology" },
  { to: "/evaluation", label: "Model evaluation", icon: ChartColumn, title: "Model evaluation" },
  { to: "/explainability", label: "Explainability", icon: ScanSearch, title: "Grad-CAM explainability" },
  { to: "/inference", label: "Classification", icon: Brain, title: "Run the classifier" },
  { to: "/limitations", label: "Limitations", icon: TriangleAlert, title: "Limitations" },
  { to: "/reproducibility", label: "Reproducibility", icon: Repeat, title: "Reproducibility" },
];

function Topbar({ theme, setTheme }) {
  const { pathname } = useLocation();
  const current = NAV.find((item) => (item.end ? pathname === item.to : pathname.startsWith(item.to))) || NAV[0];
  return (
    <header className="topbar">
      <div>
        <span>BIOMEDICAL AI</span>
        <i>/</i>
        <strong>{current.title}</strong>
      </div>
      <button className="theme" aria-label="Toggle theme" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
        {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
      </button>
    </header>
  );
}

export default function App() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("neuro-theme") || "dark";
    } catch {
      return "dark";
    }
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("neuro-theme", theme);
    } catch {
      /* storage unavailable: theme simply is not remembered */
    }
  }, [theme]);

  return (
    <BrowserRouter>
      <div className="neuro-shell">
        <aside className="sidebar">
          <NavLink className="brand" to="/">
            <span className="brand-mark"><Brain size={23} /></span>
            <span>
              <b>NEUROVISION</b>
              <small>BRAIN MRI / DL</small>
            </span>
          </NavLink>
          <div className="side-label">DASHBOARD</div>
          <nav className="side-nav" aria-label="Pages">
            {NAV.map(({ to, label, icon: Icon, end }) => (
              <NavLink key={to} to={to} end={end} className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}>
                <Icon size={16} /> {label}
              </NavLink>
            ))}
          </nav>
          <div className="side-bottom">
            <ShieldCheck size={14} /> Educational only · Not a diagnostic tool
          </div>
        </aside>
        <main className="main">
          <Topbar theme={theme} setTheme={setTheme} />
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/methodology" element={<Methodology />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/explainability" element={<Explainability />} />
            <Route path="/inference" element={<Inference />} />
            <Route path="/limitations" element={<Limitations />} />
            <Route path="/reproducibility" element={<Reproducibility />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
          <footer>
            <span><ShieldAlert size={11} /> Research and portfolio demonstration · not clinically validated</span>
            <span><i>MobileNetV2</i> · 4-class image-level classification · Grad-CAM</span>
          </footer>
        </main>
      </div>
    </BrowserRouter>
  );
}
