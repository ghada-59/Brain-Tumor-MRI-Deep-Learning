import { TriangleAlert, CircleCheck, CircleX } from "lucide-react";
import provenance from "../data/gradcam_examples.json";
import { displayClassName, shortHash } from "../lib/format.js";

const FIGURE_SRC = "/results/gradcam_examples.png";
// The composite figure is 1080 x 1440 px: a title band, then four equal rows
// (Input | Grad-CAM | Overlay), one per class. These offsets only crop the real image.
const FIGURE_HEIGHT = 1440;
const ROW_TOP = 52;
const ROW_HEIGHT = 347;
const ROW_STEP = 346;

function rowPosition(index) {
  const top = ROW_TOP + index * ROW_STEP;
  return `0% ${(top / (FIGURE_HEIGHT - ROW_HEIGHT)) * 100}%`;
}

export default function Explainability() {
  const rows = provenance.rows;
  const hits = rows.filter((row) => row.true_label === row.predicted_label).length;

  return (
    <div className="page">
      <div className="section-head">
        <span>01</span>
        <h2>Grad-CAM Visual Explanations</h2>
        <small>ONE REAL HELD-OUT IMAGE PER CLASS</small>
      </div>

      <section className="card gradcam-hero">
        <img
          src={FIGURE_SRC}
          width="1080"
          height="1440"
          alt="Grad-CAM figure: for a glioma, meningioma, no-tumor and pituitary test image, the input MRI, the Grad-CAM heatmap for the predicted class and an overlay, with the true and predicted label under each row"
          className="gradcam-figure"
        />
        <p className="card-text">
          Each row is one genuine image from the held-out <code>Testing/</code> folder: input, Grad-CAM heatmap for the
          predicted class, and overlay. Images are the first of each class in sorted order, not hand-picked. Generated
          from the evaluated checkpoint (SHA-256 <code>{shortHash(provenance.model_sha256)}</code>);{" "}
          {hits} of {rows.length} are classified correctly.
        </p>
      </section>

      <div className="section-head">
        <span>02</span>
        <h2>The four examples</h2>
        <small>TRUE LABEL · PREDICTION · SCORES</small>
      </div>
      <div className="gradcam-grid">
        {rows.map((row, index) => {
          const correct = row.true_label === row.predicted_label;
          const ranked = Object.entries(row.scores).sort((a, b) => b[1] - a[1]);
          return (
            <section className={"card gradcam-card" + (correct ? "" : " wrong")} key={row.image_sha256}>
              <div
                className="gradcam-crop"
                role="img"
                aria-label={`Input, Grad-CAM and overlay for the ${displayClassName(row.true_label)} test image`}
                style={{ backgroundImage: `url(${FIGURE_SRC})`, backgroundPosition: rowPosition(index) }}
              />
              <div className="gradcam-meta">
                <div className="gradcam-verdict">
                  {correct ? <CircleCheck size={15} /> : <CircleX size={15} />}
                  <b>{correct ? "Correct" : "Misclassified"}</b>
                </div>
                <dl>
                  <dt>True label</dt>
                  <dd>{displayClassName(row.true_label)}</dd>
                  <dt>Predicted</dt>
                  <dd>
                    {displayClassName(row.predicted_label)} ({row.scores[row.predicted_label].toFixed(3)})
                  </dd>
                </dl>
                <ul className="gradcam-scores">
                  {ranked.map(([name, score]) => (
                    <li key={name}>
                      <span>{displayClassName(name)}</span>
                      <i style={{ width: `${Math.max(score * 100, 1)}%` }} />
                      <em>{score.toFixed(3)}</em>
                    </li>
                  ))}
                </ul>
                <small className="gradcam-hash">image sha256 {shortHash(row.image_sha256, 10)}</small>
              </div>
            </section>
          );
        })}
      </div>

      <section className="card limits-card">
        <div className="limit-row">
          <TriangleAlert size={16} />
          <p>
            Grad-CAM is an <b>approximate explanation of model evidence</b>, not a segmentation mask and not a
            validated tumour localisation. The red box is a threshold-derived rectangle, not a detection.
          </p>
        </div>
        <div className="limit-row">
          <TriangleAlert size={16} />
          <p>
            In these four examples the highlighted regions <b>do not reliably coincide with visible lesions</b>: the
            glioma map (a wrong prediction) lies on the neck, the meningioma map beside the bright mass. Four images
            say nothing about general quality. Scores are uncalibrated softmax outputs, not diagnoses.
          </p>
        </div>
      </section>
    </div>
  );
}
