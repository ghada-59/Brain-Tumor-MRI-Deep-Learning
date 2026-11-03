export function displayClassName(name) {
  return name === "notumor" ? "No tumor" : name.charAt(0).toUpperCase() + name.slice(1);
}

export function pct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function shortHash(hash, length = 12) {
  if (!hash) return "unavailable";
  return `${hash.slice(0, length)}…`;
}

/**
 * Compares the live inference model's hash (from /api/status) against the
 * checkpoint hash the evaluation figures describe. Never assume they match
 * just because both are "a MobileNetV2 checkpoint" -- the project can have,
 * and currently does have, more than one trained checkpoint on disk.
 */
export function modelIdentityState(liveSha256, evaluatedSha256) {
  if (!liveSha256 || !evaluatedSha256) return "unknown";
  return liveSha256 === evaluatedSha256 ? "match" : "mismatch";
}
