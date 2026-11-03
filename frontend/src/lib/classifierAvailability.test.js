import { test } from "node:test";
import assert from "node:assert/strict";
import { classifierAvailability, predictErrorMessage } from "./classifierAvailability.js";

test("a checkpoint that failed to load cannot be used, even though the file exists", () => {
  const status = { model_available: false, model_state: "load_failed", model_error: "Corrupt file.", expected_model: "x.keras" };
  const result = classifierAvailability("online", status);
  assert.equal(result.canRun, false);
  assert.match(result.problem, /Corrupt file\./);
});

test("ready and lazily-loaded models can run", () => {
  assert.equal(classifierAvailability("online", { model_state: "ready" }).canRun, true);
  assert.equal(classifierAvailability("online", { model_state: "not_loaded" }).canRun, true);
});

test("missing weights and an offline API block the button", () => {
  assert.equal(classifierAvailability("online", { model_state: "missing" }).canRun, false);
  assert.equal(classifierAvailability("offline", null).canRun, false);
  assert.equal(classifierAvailability("checking", null).canRun, false);
});

test("older API versions without model_state fall back to model_available", () => {
  assert.equal(classifierAvailability("online", { model_available: true }).canRun, true);
  assert.equal(classifierAvailability("online", { model_available: false }).canRun, false);
});

test("API errors become one readable sentence, never [object Object]", () => {
  assert.equal(predictErrorMessage(415, "Upload a PNG, JPG, BMP or WEBP image."), "Unsupported file type: Upload a PNG, JPG, BMP or WEBP image.");
  assert.equal(predictErrorMessage(422, [{ msg: "Field required" }]), "Request failed (HTTP 422): Field required");
  assert.equal(predictErrorMessage(500, undefined), "Server error.");
});
