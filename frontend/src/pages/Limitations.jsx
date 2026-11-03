import { ShieldAlert } from "lucide-react";

const LIMITATIONS = [
  {
    title: "Not a medical device",
    body: "This project is for education and portfolio demonstration. It is not a medical device and must not be used to diagnose or exclude a brain tumour, or to guide treatment.",
  },
  {
    title: "Not clinically validated",
    body: "Neither the MobileNetV2 classifier nor its Grad-CAM explanations have been validated against expert radiological ground truth or evaluated in a clinical setting.",
  },
  {
    title: "Classification is not segmentation or localization",
    body: "The model outputs one label per image from four dataset classes. It does not predict a tumour contour, bounding box, or center, regardless of what a Grad-CAM heatmap or activation box appears to highlight.",
  },
  {
    title: "Grad-CAM is an approximate explanation",
    body: "Saliency/attention visualizations show which regions influenced the model's output. They can highlight anatomy or scan artifacts as easily as diagnostically relevant tissue, and have not been checked against lesion masks.",
  },
  {
    title: "Test-set reuse",
    body: "The same Testing/ folder has already been consulted in earlier, historical model comparisons in this project. It should not be treated as a pristine, never-seen independent benchmark.",
  },
  {
    title: "No patient identifiers",
    body: "This dataset provides no patient or examination identifiers. Patient-level independence between training and test images cannot be established by any method used in this project, including the near-duplicate audit.",
  },
  {
    title: "Probable same-patient overlap between training and test",
    body: "Because slices of the same patient can appear on both sides of the split and a perceptual-hash screen cannot detect different slices of one patient, the reported accuracy is slice-level and probably optimistic. It must not be read as an estimate of performance on new patients or other scanners.",
  },
  {
    title: "Near-duplicate audit is not exhaustive",
    body: "The leakage control excludes training images that perceptually hash-matched a test image at the strictest, zero-distance threshold. It does not prove the absence of every possible near-duplicate or other leakage mechanism.",
  },
  {
    title: "T-PAL/Transformer component is untrained",
    body: "The repository's separate PyTorch Transformer-style classifier has never been trained end to end. No accuracy, F1, or any other evidence-based result exists for it, and it must not be presented as a validated alternative to the MobileNetV2 result above.",
  },
  {
    title: "Single training run",
    body: "The reported result comes from one training run at one random seed. No repeated-seed variance estimate exists, so part of the difference from other configurations could reflect run-to-run variation rather than only the dataset change.",
  },
];

export default function Limitations() {
  return (
    <div className="page">
      <div className="section-head">
        <span>01</span>
        <h2>Limitations</h2>
        <small>READ BEFORE CITING ANY RESULT ABOVE</small>
      </div>
      <div className="limits-list">
        {LIMITATIONS.map((item) => (
          <section className="card limit-card" key={item.title}>
            <div className="limit-row">
              <ShieldAlert size={18} />
              <div>
                <h3>{item.title}</h3>
                <p>{item.body}</p>
              </div>
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
