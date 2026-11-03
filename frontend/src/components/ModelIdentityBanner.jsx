import { Link } from "react-router-dom";
import { Check, HelpCircle, TriangleAlert } from "lucide-react";
import results from "../data/results.json";
import { modelIdentityState, shortHash } from "../lib/format.js";

const EVALUATED_SHA256 = results.selected_experiment.checkpoint_sha256;

/**
 * Makes explicit whether the model actually serving predictions right now
 * is the same checkpoint described on the Model Evaluation page. The two
 * can differ -- this project keeps more than one trained checkpoint on
 * disk, and the API's default resolver does not necessarily point at the
 * one used for the reported evaluation. Never implied to match by default.
 */
export default function ModelIdentityBanner({ apiStatus }) {
  if (!apiStatus || !apiStatus.model_available) return null;

  const liveSha256 = apiStatus.model_sha256;
  const state = modelIdentityState(liveSha256, EVALUATED_SHA256);

  if (state === "match") {
    return (
      <div className="scope-banner identity-ok">
        <Check size={18} />
        <p>
          <b>Live model matches the evaluated checkpoint.</b> Predictions below come from the same checkpoint
          (<code>{shortHash(liveSha256, 10)}</code>) whose metrics appear on{" "}
          <Link to="/evaluation">Model Evaluation</Link>.
        </p>
      </div>
    );
  }

  if (state === "mismatch") {
    return (
      <div className="alert warm">
        <TriangleAlert size={17} />
        <span>
          <b>The live inference model is not the checkpoint described on the Model Evaluation page.</b> This API is
          currently serving a different checkpoint (<code>{shortHash(liveSha256, 10)}</code> vs. the evaluated{" "}
          <code>{shortHash(EVALUATED_SHA256, 10)}</code>). Predictions below are real model outputs, but their
          accuracy has not been independently reported on this dashboard — do not assume they match the 83.46%
          accuracy / 82.85% macro-F1 shown on <Link to="/evaluation">Model Evaluation</Link>.
        </span>
      </div>
    );
  }

  return (
    <div className="scope-banner">
      <HelpCircle size={18} />
      <p>
        Model identity could not be verified against the evaluated checkpoint (the running API does not report a
        checkpoint hash, or no evaluated checkpoint hash is available). Treat predictions below as coming from an
        unverified checkpoint rather than assuming they match <Link to="/evaluation">Model Evaluation</Link>.
      </p>
    </div>
  );
}
