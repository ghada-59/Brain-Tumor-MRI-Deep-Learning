import results from "../data/results.json";
import StatCard from "../components/StatCard.jsx";
import PerClassBarChart from "../components/PerClassBarChart.jsx";
import ConfusionMatrixTable from "../components/ConfusionMatrixTable.jsx";
import TrainingCurveChart from "../components/TrainingCurveChart.jsx";
import { pct } from "../lib/format.js";

export default function Evaluation() {
  const { test_metrics: metrics, classification_report: report, class_names: classNames, confusion_matrix: confusion, historical_comparison: historical } =
    results;

  return (
    <div className="page">
      <div className="section-head">
        <span>01</span>
        <h2>Aggregate metrics</h2>
        <small>SINGLE EVALUATION · n = {metrics.test_image_count}</small>
      </div>
      <p className="lead-text">
        Evaluated exactly once on the <code>Testing/</code> folder, after training and checkpoint selection (by
        validation macro-F1) were already finalized. This number was not used to choose epochs, fine-tuning depth, or
        between candidate checkpoints.
      </p>
      <div className="stat-grid">
        <StatCard label="Accuracy" value={pct(metrics.test_accuracy)} />
        <StatCard label="Macro-F1" value={pct(metrics.test_macro_f1)} />
        <StatCard label="Weighted-F1" value={pct(metrics.test_weighted_f1)} />
        <StatCard label="Test images" value={metrics.test_image_count} />
      </div>

      <div className="section-head">
        <span>02</span>
        <h2>Per-class performance</h2>
        <small>PRECISION · RECALL · F1</small>
      </div>
      <section className="card">
        <PerClassBarChart report={report} classNames={classNames} />
      </section>

      <div className="section-head">
        <span>03</span>
        <h2>Confusion matrix</h2>
        <small>ROWS: TRUE · COLUMNS: PREDICTED</small>
      </div>
      <section className="card">
        <ConfusionMatrixTable matrix={confusion.matrix} classNames={classNames} />
        <p className="card-text">
          The dominant error is <b>glioma predicted as meningioma</b> (95 of 382 glioma images) — a confusion
          direction consistent with the two tumor types' overlapping MRI appearance, not something the leakage
          controls applied to this dataset were designed to fix.
        </p>
      </section>

      <div className="section-head">
        <span>04</span>
        <h2>Training and validation curves</h2>
        <small>FROZEN BACKBONE → FINE-TUNING</small>
      </div>
      <section className="card chart-grid">
        <TrainingCurveChart
          frozenHistory={results.training_history_frozen}
          finetuneHistory={results.training_history_finetune}
          metricKey="val_macro_f1"
          label="Validation macro-F1 (checkpoint-selection metric)"
        />
        <TrainingCurveChart
          frozenHistory={results.training_history_frozen}
          finetuneHistory={results.training_history_finetune}
          metricKey="val_loss"
          label="Validation loss"
        />
      </section>

      <div className="section-head">
        <span>05</span>
        <h2>Historical comparison (context only)</h2>
        <small>NOT THE PRIMARY RESULT</small>
      </div>
      <section className="card">
        <table className="data-table">
          <caption>{historical.note}</caption>
          <thead>
            <tr>
              <th scope="col">Configuration</th>
              <th scope="col">Test accuracy</th>
              <th scope="col">Macro-F1</th>
            </tr>
          </thead>
          <tbody>
            {historical.configs.map((row) => (
              <tr key={row.label}>
                <th scope="row">{row.label}</th>
                <td>{pct(row.test_accuracy)}</td>
                <td>{pct(row.test_macro_f1)}</td>
              </tr>
            ))}
            <tr className="current-row">
              <th scope="row">Leakage-controlled (current, primary result)</th>
              <td>{pct(metrics.test_accuracy)}</td>
              <td>{pct(metrics.test_macro_f1)}</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
