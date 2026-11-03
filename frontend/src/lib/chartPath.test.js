import { test } from "node:test";
import assert from "node:assert/strict";
import { linePath } from "./chartPath.js";

test("every series starts with a move-to command, even when it starts mid-chart", () => {
  // Regression: the fine-tuning curve is drawn after the frozen phase, and its
  // path used to start with "L", which is invalid SVG and was not rendered.
  assert.equal(linePath([[400, 47.2], [436, 50]]), "M400.0,47.2 L436.0,50.0");
});

test("missing values break the line instead of emitting NaN", () => {
  assert.equal(linePath([[0, 1], [10, Number.NaN], [20, 3], [30, 4]]), "M0.0,1.0 M20.0,3.0 L30.0,4.0");
});

test("no drawable point gives an empty path", () => {
  assert.equal(linePath([]), "");
  assert.equal(linePath([[1, Number.NaN]]), "");
});
