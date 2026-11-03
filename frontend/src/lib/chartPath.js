/**
 * SVG path for one polyline. Every series starts with a move-to ("M") command;
 * a path that starts with "L" is invalid SVG and browsers draw nothing for it.
 * Points whose coordinates are not finite (a missing metric in a history row)
 * break the line instead of producing "NaN" coordinates.
 *
 * @param {Array<[number, number]>} points x/y pairs in SVG user units
 * @returns {string} the value for a <path d="..."> attribute ("" when no point is drawable)
 */
export function linePath(points) {
  const commands = [];
  let penDown = false;
  for (const [x, y] of points) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      penDown = false;
      continue;
    }
    commands.push(`${penDown ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`);
    penDown = true;
  }
  return commands.join(" ");
}
