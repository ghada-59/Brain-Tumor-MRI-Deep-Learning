import { Link } from "react-router-dom";
import { ArrowRight, ShieldAlert } from "lucide-react";
import results from "../data/results.json";
import StatCard from "../components/StatCard.jsx";
import { pct } from "../lib/format.js";

export default function Overview() {
  const { test_metrics: metrics, selected_experiment: experiment, classification_report: report } = results;

  return (
    <div className="page">
      <section className="hero">
        <div>
          <span className="eyebrow">TRANSFER LEARNING · MOBILENETV2 · LEAKAGE-CONTROLLED</span>
          <h1>
            Brain MRI classification<span>.</span>
          </h1>
          <p>
            A four-class image-level classifier (<code>glioma</code>, <code>meningioma</code>, <code>notumor</code>,{" "}
            <code>pituitary</code>) built around a MobileNetV2 transfer-learning backbone. The headline result below
            comes from a dataset copy where 338 training images identified as near-duplicates of held-out test
            images were excluded, with a group-aware train/validation split and training-partition-only class
            weights.
          </p>
        </div>
      </section>

      <div className="scope-banner">
        <ShieldAlert size={18} />
        <p>
          <b>Research and portfolio demonstration only.</b> Not a medical device. Must not be used to diagnose or
          exclude a brain tumour, or to guide treatment. Scores are model outputs, not clinically calibrated
          confidence.
        </p>
      </div>

      <div className="section-head">
        <span>01</span>
        <h2>Primary result</h2>
        <small>{experiment.label.toUpperCase()}</small>
      </div>

      <div className="stat-grid">
        <StatCard label="Test accuracy" value={pct(metrics.test_accuracy)} hint={`n = ${metrics.test_image_count}`} />
        <StatCard label="Macro-F1" value={pct(metrics.test_macro_f1)} hint="unweighted across 4 classes" />
        <StatCard label="Weighted-F1" value={pct(metrics.test_weighted_f1)} hint="weighted by class support" />
        <StatCard
          label="Weakest class recall"
          value={pct(report.glioma.recall)}
          hint="glioma — see Limitations"
          tone="warn"
        />
      </div>

      <div className="split-grid">
        <section className="card">
          <div className="card-head">
            <div>
              <small>MAIN TECHNICAL CONTRIBUTION</small>
              <h3>Dataset-integrity auditing, not just a trained model</h3>
            </div>
          </div>
          <p className="card-text">
            A systematic perceptual-hash audit found that 59.0% of the <code>notumor</code> test images have a
            near-certain visual duplicate in the training data. The model above was retrained on a dataset copy that
            specifically excludes those 338 training images, with a group-aware split so the matched image clusters
            cannot be separated across train/validation by chance. See{" "}
            <Link to="/methodology">Data &amp; methodology</Link> for the full numbers.
          </p>
        </section>
        <section className="card">
          <div className="card-head">
            <div>
              <small>PRINCIPAL LIMITATION</small>
              <h3>Test-set reuse and unverifiable patient independence</h3>
            </div>
          </div>
          <p className="card-text">
            The same <code>Testing/</code> folder has already been consulted in earlier comparisons — it is not a
            pristine, never-seen benchmark. This dataset has no patient identifiers, so patient-level independence
            cannot be established by any method used here. See <Link to="/limitations">Limitations</Link> for the
            complete list.
          </p>
        </section>
      </div>

      <div className="nav-cards">
        <Link className="nav-card" to="/evaluation">
          <span>Model evaluation</span>
          <ArrowRight size={16} />
        </Link>
        <Link className="nav-card" to="/inference">
          <span>Run the classifier</span>
          <ArrowRight size={16} />
        </Link>
        <Link className="nav-card" to="/reproducibility">
          <span>Reproduce this result</span>
          <ArrowRight size={16} />
        </Link>
      </div>
    </div>
  );
}
