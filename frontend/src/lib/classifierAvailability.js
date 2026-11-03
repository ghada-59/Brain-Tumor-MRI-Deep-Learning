/**
 * Whether the Classification page may send an image, and what to tell the user.
 * Uses /api/status `model_state` (API >= 1.1); falls back to `model_available`
 * for older API versions.
 *
 * @param {"checking"|"online"|"offline"} apiState
 * @param {object|null} status
 * @returns {{canRun: boolean, label: string, problem: string|null}}
 */
export function classifierAvailability(apiState, status) {
  if (apiState === "checking") return { canRun: false, label: "Checking service…", problem: null };
  if (apiState !== "online" || !status) return { canRun: false, label: "API offline", problem: null };
  const where = status.expected_model ? ` (${status.expected_model})` : "";
  switch (status.model_state) {
    case "ready":
      return { canRun: true, label: "Model loaded and ready", problem: null };
    case "not_loaded":
      return { canRun: true, label: "Model will load on the first request", problem: null };
    case "missing":
      return {
        canRun: false,
        label: "Model weights not found",
        problem: `Model weights are not on this machine${where}. They are not distributed with the repository; train locally or set BRAIN_TUMOR_MODEL_PATH.`,
      };
    case "load_failed":
      return {
        canRun: false,
        label: "Model failed to load",
        problem: `The checkpoint${where} could not be used: ${status.model_error || "unknown error"} See the API log for details.`,
      };
    default:
      return status.model_available
        ? { canRun: true, label: "Model weights found", problem: null }
        : { canRun: false, label: "Local model weights required", problem: `Model weights are not on this machine${where}.` };
  }
}

const STATUS_PREFIX = {
  400: "Image rejected",
  413: "File too large",
  415: "Unsupported file type",
  500: "Server error",
  503: "Model unavailable",
};

/**
 * One readable sentence for a failed /api/predict call.
 * `detail` is the JSON `detail` field (a string from this API; a list from older FastAPI 422s).
 */
export function predictErrorMessage(httpStatus, detail) {
  let text = "";
  if (typeof detail === "string") text = detail;
  else if (Array.isArray(detail)) text = detail.map((item) => item && item.msg).filter(Boolean).join("; ");
  const prefix = STATUS_PREFIX[httpStatus] || `Request failed (HTTP ${httpStatus})`;
  return text ? `${prefix}: ${text}` : `${prefix}.`;
}
