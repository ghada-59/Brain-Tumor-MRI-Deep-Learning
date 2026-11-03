import { useEffect, useRef, useState } from "react";
import { Check, CircleAlert, FileImage, LoaderCircle, ShieldCheck, Upload, X } from "lucide-react";
import { displayClassName } from "../lib/format.js";
import ModelIdentityBanner from "../components/ModelIdentityBanner.jsx";
import { classifierAvailability, predictErrorMessage } from "../lib/classifierAvailability.js";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const ACCEPTED_EXTENSIONS = /\.(png|jpe?g|webp|bmp)$/i;
const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;

function useApiStatus() {
  const [state, setState] = useState("checking");
  const [status, setStatus] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/status`)
      .then((response) => {
        if (!response.ok) throw new Error();
        return response.json();
      })
      .then((data) => {
        if (cancelled) return;
        setStatus(data);
        setState("online");
      })
      .catch(() => {
        if (!cancelled) setState("offline");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return [state, status];
}

export default function Inference() {
  const fileInputRef = useRef(null);
  const [apiState, apiStatus] = useApiStatus();
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const availability = classifierAvailability(apiState, apiStatus);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function chooseFile(next) {
    setError("");
    setResult(null);
    if (!next) return;
    if (!ACCEPTED_EXTENSIONS.test(next.name)) {
      setError("Choose a PNG, JPG, WEBP or BMP MRI image.");
      return;
    }
    if (next.size > MAX_UPLOAD_BYTES) {
      setError("Maximum upload size is 15 MB.");
      return;
    }
    setFile(next);
  }

  async function runInference() {
    if (!file) {
      setError("Select an MRI image first.");
      return;
    }
    setBusy(true);
    setError("");
    setResult(null);
    const body = new FormData();
    body.append("file", file);
    try {
      let response;
      try {
        response = await fetch(`${API_BASE}/api/predict`, { method: "POST", body });
      } catch {
        throw new Error(`Could not connect to the inference API at ${API_BASE}.`);
      }
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(predictErrorMessage(response.status, data.detail));
      setResult(data);
    } catch (err) {
      setError(err.message || "Inference failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <div className="side-status standalone">
        <span className={"led " + apiState} />
        <div>
          <b>{apiState === "online" ? "API connected" : apiState === "checking" ? "Checking service…" : "API offline"}</b>
          <small>{availability.label}</small>
        </div>
      </div>

      <ModelIdentityBanner apiStatus={apiStatus} />

      <div className="scope-banner">
        <ShieldCheck size={18} />
        <p>
          <b>Research demonstration only.</b> This runs the project's actual FastAPI inference path against a local
          model — predictions are real model outputs, never simulated. Scores are not a diagnosis.
        </p>
      </div>

      {error && (
        <div className="alert">
          <CircleAlert size={17} />
          <span>{error}</span>
          <button onClick={() => setError("")} aria-label="Dismiss error">
            <X size={15} />
          </button>
        </div>
      )}
      {apiState === "offline" && (
        <div className="alert">
          <CircleAlert size={17} />
          <span>
            The Python inference API is unreachable at <code>{API_BASE}</code>. Start it locally
            (<code>python -m uvicorn api:app --host 127.0.0.1 --port 8000</code>) and reload this page. The Model
            Evaluation page still works from saved results without the API.
          </span>
        </div>
      )}
      {availability.problem && (
        <div className="alert warm">
          <FileImage size={17} />
          <span>{availability.problem}</span>
        </div>
      )}

      <div className="section-head">
        <span>01</span>
        <h2>MRI input</h2>
        <small>UPLOAD · PREVIEW · INFER</small>
      </div>

      <div className="work-grid">
        <section className="card">
          <div className="card-head">
            <div>
              <small>INPUT IMAGE</small>
              <h3>Upload a brain MRI</h3>
            </div>
          </div>
          <input
            className="hidden"
            ref={fileInputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.webp,.bmp"
            aria-label="Upload an MRI image"
            onChange={(event) => chooseFile(event.target.files && event.target.files[0])}
          />
          {previewUrl ? (
            <>
              <div className="image-preview">
                <img src={previewUrl} alt="Selected MRI preview" />
              </div>
              <div className="file-row">
                <div>
                  <b>{file.name}</b>
                  <small>{(file.size / 1024).toFixed(0)} KB · preview ready</small>
                </div>
                <button
                  onClick={() => {
                    setFile(null);
                    setResult(null);
                    if (fileInputRef.current) fileInputRef.current.value = "";
                  }}
                  aria-label="Remove image"
                >
                  <X size={15} />
                </button>
              </div>
              <p className="loaded">
                <Check size={14} /> Image loaded locally (never uploaded anywhere except your own API call)
              </p>
              <button className="change" onClick={() => fileInputRef.current && fileInputRef.current.click()}>
                <Upload size={14} /> Replace image
              </button>
            </>
          ) : (
            <button
              className="dropzone"
              onClick={() => fileInputRef.current && fileInputRef.current.click()}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                chooseFile(event.dataTransfer.files && event.dataTransfer.files[0]);
              }}
            >
              <span className="upload-symbol">
                <Upload size={22} />
              </span>
              <b>Drop an MRI image here</b>
              <small>or browse local files</small>
              <span>PNG · JPG · WEBP · BMP / 15 MB MAX</span>
            </button>
          )}
          <button className="run" disabled={!file || busy || !availability.canRun} onClick={runInference}>
            {busy ? (
              <>
                <LoaderCircle className="spin" size={16} /> Running model…
              </>
            ) : (
              <>
                Run classification <span>→</span>
              </>
            )}
          </button>
        </section>

        <aside className="card class-info">
          <div className="card-head">
            <div>
              <small>MODEL INFORMATION</small>
              <h3>Four dataset classes</h3>
            </div>
          </div>
          {["glioma", "meningioma", "notumor", "pituitary"].map((name, index) => (
            <div className="class-row" key={name}>
              <span className={"class-number n" + index}>{"0" + (index + 1)}</span>
              <div>
                <b>{displayClassName(name)}</b>
                <small>Dataset label · {name}</small>
              </div>
            </div>
          ))}
          <div className="model-fact">
            <span>BACKBONE</span>
            <b>MobileNetV2</b>
            <span>INPUT SHAPE</span>
            <b>224 × 224 × 3</b>
            <span>OUTPUT TYPE</span>
            <b>Image-level class scores</b>
          </div>
        </aside>
      </div>

      <div className="section-head results-head">
        <span>02</span>
        <h2>Model output</h2>
        <small>{result ? "INFERENCE COMPLETE" : "WAITING FOR INFERENCE"}</small>
      </div>
      {result ? (
        <section className="result-card">
          <div className="prediction-head">
            <img src={previewUrl} alt="Uploaded image (display preview)" />
            <div>
              <small>TOP-RANKED DATASET CLASS</small>
              <h3>{displayClassName(result.top_class)}</h3>
              <span>Model output · not a diagnosis</span>
            </div>
            <span className="completed">
              <Check size={14} /> COMPLETE
            </span>
          </div>
          <div className="score-list">
            {result.classes.map((item, index) => (
              <div className="score-row" key={item.label}>
                <div className="score-label">
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <b>{displayClassName(item.label)}</b>
                  <strong>{(item.score * 100).toFixed(2)}%</strong>
                </div>
                <div className="score-track">
                  <span style={{ width: `${Math.max(0, Math.min(100, item.score * 100))}%` }} />
                </div>
              </div>
            ))}
          </div>
          {result.model && (
            <p className="card-text">
              Checkpoint <code>{String(result.model.sha256 || "unknown").slice(0, 12)}…</code> ({result.model.variant})
              {result.preprocessing ? ` · decoded with ${result.preprocessing.decoder}` : ""}
              {typeof result.latency_ms === "number" ? ` · ${result.latency_ms} ms` : ""}
            </p>
          )}
          <div className="scope-banner result-warning">
            <ShieldCheck size={17} />
            <p>{result.note}</p>
          </div>
        </section>
      ) : (
        <div className="empty">
          <FileImage size={28} />
          <b>No inference results yet</b>
          <span>
            Upload an MRI and run the trained model. If model weights are missing, the interface reports that
            limitation rather than fabricating scores.
          </span>
        </div>
      )}
    </div>
  );
}
