import results from "../data/results.json";
import { shortHash } from "../lib/format.js";

function Field({ label, value }) {
  return (
    <div className="field-row">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

// What the recorded commit says about the code that ran. A record written before the
// check existed has no value: it is reported as unknown, never as clean.
export function sourceStateLabel(dirty) {
  if (dirty === true) return "Yes — the commit alone does not identify the code that ran";
  if (dirty === false) return "None — the commit identifies the code that ran";
  return "Not recorded for this run — the commit may not match the code that ran (see README)";
}

export default function Reproducibility() {
  const e = results.selected_experiment;

  return (
    <div className="page">
      <div className="section-head">
        <span>01</span>
        <h2>Selected checkpoint</h2>
      </div>
      <section className="card mono-card">
        <Field label="Architecture" value={e.architecture} />
        <Field label="Checkpoint path" value={e.checkpoint_path} />
        <Field label="Checkpoint SHA-256" value={<code title={e.checkpoint_sha256}>{shortHash(e.checkpoint_sha256, 20)}</code>} />
        <Field label="Git commit" value={e.git_commit || "unavailable"} />
        <Field label="Uncommitted source changes" value={sourceStateLabel(e.git_source_dirty)} />
      </section>

      <div className="section-head">
        <span>02</span>
        <h2>Training configuration</h2>
      </div>
      <section className="card mono-card">
        <Field label="Dataset" value={e.data_dir} />
        <Field label="Group manifest" value={e.group_manifest || "not used"} />
        <Field label="Seed" value={e.seed} />
        <Field label="Validation split" value={e.validation_split} />
        <Field label="Image size" value={`${e.image_size[0]} × ${e.image_size[1]}`} />
        <Field label="Fine-tuned layers" value={e.fine_tune_layers} />
        <Field label="Checkpoint-selection metric" value={e.selection_metric} />
        <Field
          label="Epochs (frozen → fine-tune)"
          value={`${e.actual_epochs_frozen} of ${e.requested_epochs} requested → ${e.actual_epochs_finetune} of ${e.requested_epochs} requested`}
        />
      </section>

      <div className="section-head">
        <span>03</span>
        <h2>Class weights (training partition only)</h2>
      </div>
      <section className="card">
        <table className="data-table">
          <caption>Inverse-frequency weights computed from the realized training partition — never from validation or test labels</caption>
          <thead>
            <tr>
              <th scope="col">Class</th>
              <th scope="col">Weight</th>
            </tr>
          </thead>
          <tbody>
            {e.class_weight &&
              Object.entries(e.class_weight).map(([name, weight]) => (
                <tr key={name}>
                  <th scope="row">{name}</th>
                  <td>{weight.toFixed(4)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </section>

      <div className="section-head">
        <span>04</span>
        <h2>Software environment</h2>
      </div>
      <section className="card mono-card">
        {Object.entries(e.software_versions).map(([name, version]) => (
          <Field key={name} label={name} value={version} />
        ))}
      </section>

      <div className="section-head">
        <span>05</span>
        <h2>Reproduce this result</h2>
      </div>
      <section className="card">
        <pre className="command-block">
{`architecture      MobileNetV2 (ImageNet weights), fine-tuning last ${e.fine_tune_layers} layers
data              ${e.data_dir}  (leakage-controlled copy, see Data & methodology)
image size        ${e.image_size[0]} x ${e.image_size[1]}
validation split  ${e.validation_split}  (group-aware, seed ${e.seed})
epochs            ${e.actual_epochs_frozen} frozen + ${e.actual_epochs_finetune} fine-tuning (requested ${e.requested_epochs})
selection metric  ${e.selection_metric}
class weights     computed from the training partition only`}
        </pre>
        <p className="card-text">
          <b>Honest scope:</b> these are the recorded settings of the run. The <code>train.py</code> published in
          this repository covers the plain frozen-backbone and fine-tuning workflow; the exact training script
          variant used for this checkpoint (group manifest, class weights, macro-F1 checkpoint selection) is not
          included, so the run is documented but not one-command reproducible. The evaluated checkpoint itself is
          shipped and verifiable by SHA-256.
        </p>
      </section>
    </div>
  );
}
