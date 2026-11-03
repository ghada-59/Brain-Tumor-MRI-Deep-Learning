import results from "../data/results.json";
import { displayClassName } from "../lib/format.js";

export default function Methodology() {
  const { leakage_summary: leakage, selected_experiment: experiment, class_names: classNames } = results;
  const before = leakage.class_balance.before.Training;
  const after = leakage.class_balance.after.Training;
  const testCounts = leakage.class_balance.before.Testing;

  return (
    <div className="page">
      <div className="section-head">
        <span>01</span>
        <h2>Dataset and class counts</h2>
      </div>
      <section className="card">
        <table className="data-table">
          <caption>Training partition before/after leakage control, and Testing/ (unchanged)</caption>
          <thead>
            <tr>
              <th scope="col">Class</th>
              <th scope="col">Training (before)</th>
              <th scope="col">Training (after exclusion)</th>
              <th scope="col">Testing (unchanged)</th>
            </tr>
          </thead>
          <tbody>
            {classNames.map((name) => (
              <tr key={name}>
                <th scope="row">{displayClassName(name)}</th>
                <td>{before[name]}</td>
                <td>{after[name]}</td>
                <td>{testCounts[name]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <div className="section-head">
        <span>02</span>
        <h2>Near-duplicate leakage audit</h2>
        <small>PERCEPTUAL HASH · DISTANCE 0</small>
      </div>
      <section className="card">
        <p className="card-text">
          A perceptual-hash audit of this dataset copy found that <b>59.0% of the <code>notumor</code> test images
          (236/400)</b> have a near-certain visual duplicate in the training data — most likely repeated or
          re-exported MRI slices from the source dataset. None of these matched pairs are byte-identical; they are
          near-duplicates, not exact copies.
        </p>
        <p className="card-text">
          <b>{leakage.total_training_images_excluded} training images</b> were excluded from this experiment's
          training data because they matched a held-out test image at this threshold (
          {Object.entries(leakage.class_balance.training_images_lost_per_class)
            .map(([name, count]) => `${count} ${displayClassName(name).toLowerCase()}`)
            .join(", ")}
          ). The <code>Testing/</code> folder itself was never modified.
        </p>
        <p className="card-text">
          <b>{leakage.pure_training_components_with_2plus_members} near-duplicate groups</b> confined to the
          train/validation side were kept intact by a group-aware split (<code>{experiment.group_manifest}</code>) so
          matched images cannot land on both sides of that split by chance.
        </p>
      </section>

      <div className="section-head">
        <span>03</span>
        <h2>What this audit does not establish</h2>
      </div>
      <section className="card limits-card">
        <p className="card-text">
          This dataset has no patient identifiers — patient-level independence between training and test images
          cannot be established by any method used in this project. The distance-0 perceptual-hash audit addresses
          the specific exact-hash groups it was designed to detect; it does not prove the absence of every possible
          near-duplicate or leakage mechanism, and the same <code>Testing/</code> folder has already been consulted
          in earlier, historical comparisons — it is not a pristine, never-seen benchmark.
        </p>
      </section>
    </div>
  );
}
