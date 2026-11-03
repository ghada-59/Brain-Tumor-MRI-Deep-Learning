import { displayClassName, pct } from "../lib/format.js";

/**
 * Horizontal grouped bar chart for per-class precision/recall/F1.
 * Pure SVG + HTML, no charting dependency. Every value also has a visible
 * numeric label so the chart is not color-only, and a table alternative is
 * rendered alongside for screen readers / text-only review.
 */
export default function PerClassBarChart({ report, classNames }) {
  const metrics = [
    { key: "precision", label: "Precision", color: "var(--cyan)" },
    { key: "recall", label: "Recall", color: "var(--teal)" },
    { key: "f1-score", label: "F1-score", color: "var(--coral)" },
  ];

  return (
    <div className="bar-chart" role="img" aria-label="Per-class precision, recall and F1-score bar chart">
      <div className="bar-chart-legend">
        {metrics.map((m) => (
          <span key={m.key} className="legend-item">
            <i style={{ background: m.color }} /> {m.label}
          </span>
        ))}
      </div>
      {classNames.map((name) => {
        const row = report[name] || {};
        return (
          <div className="bar-group" key={name}>
            <div className="bar-group-label">
              <b>{displayClassName(name)}</b>
              <small>support: {row.support ?? "—"}</small>
            </div>
            <div className="bar-group-bars">
              {metrics.map((m) => (
                <div className="bar-row" key={m.key}>
                  <span className="bar-row-name">{m.label}</span>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{ width: `${Math.max(0, Math.min(100, (row[m.key] || 0) * 100))}%`, background: m.color }}
                    />
                  </div>
                  <span className="bar-row-value">{pct(row[m.key], 1)}</span>
                </div>
              ))}
            </div>
          </div>
        );
      })}
      <table className="data-table">
        <caption>Per-class precision, recall, F1-score and support (exact values)</caption>
        <thead>
          <tr>
            <th scope="col">Class</th>
            <th scope="col">Precision</th>
            <th scope="col">Recall</th>
            <th scope="col">F1-score</th>
            <th scope="col">Support</th>
          </tr>
        </thead>
        <tbody>
          {classNames.map((name) => {
            const row = report[name] || {};
            return (
              <tr key={name}>
                <th scope="row">{displayClassName(name)}</th>
                <td>{pct(row.precision, 2)}</td>
                <td>{pct(row.recall, 2)}</td>
                <td>{pct(row["f1-score"], 2)}</td>
                <td>{row.support ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
