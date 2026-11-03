import { linePath } from "../lib/chartPath.js";

/**
 * Small hand-rolled SVG line chart (no charting library) for one metric
 * across training epochs, split into the frozen-backbone and fine-tuning
 * phases. Includes a visible axis, a legend, and numeric endpoints so the
 * chart is understandable without relying on color alone.
 */
export default function TrainingCurveChart({ frozenHistory, finetuneHistory, metricKey, label }) {
  const frozenValues = frozenHistory.map((row) => Number(row[metricKey]));
  const finetuneValues = finetuneHistory.map((row) => Number(row[metricKey]));
  const combined = [...frozenValues, ...finetuneValues];
  const finite = combined.filter(Number.isFinite);
  if (finite.length === 0) return null;

  const width = 560;
  const height = 180;
  const padding = { top: 12, right: 16, bottom: 28, left: 40 };
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const range = max - min || 1;

  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const totalEpochs = combined.length;

  function pathFor(values, offset) {
    return linePath(
      values.map((value, index) => [
        padding.left + ((offset + index) / Math.max(1, totalEpochs - 1)) * plotWidth,
        padding.top + plotHeight - ((value - min) / range) * plotHeight,
      ])
    );
  }

  const splitX = padding.left + ((frozenValues.length - 1) / Math.max(1, totalEpochs - 1)) * plotWidth;

  return (
    <figure className="line-chart">
      <figcaption>{label}</figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label} across training epochs`}>
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} className="axis-line" />
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} className="axis-line" />
        {frozenValues.length > 0 && finetuneValues.length > 0 && (
          <line x1={splitX} y1={padding.top} x2={splitX} y2={height - padding.bottom} className="phase-split" />
        )}
        <path d={pathFor(frozenValues, 0)} className="curve frozen" fill="none" />
        <path d={pathFor(finetuneValues, frozenValues.length)} className="curve finetune" fill="none" />
        <text x={padding.left} y={14} className="axis-label">{max.toFixed(2)}</text>
        <text x={padding.left} y={height - padding.bottom - 2} className="axis-label">{min.toFixed(2)}</text>
        <text x={padding.left} y={height - 8} className="axis-label">epoch 0</text>
        <text x={width - padding.right} y={height - 8} className="axis-label" textAnchor="end">
          epoch {totalEpochs - 1}
        </text>
      </svg>
      <div className="line-chart-legend">
        <span><i className="swatch frozen" /> Frozen backbone ({frozenValues.length} epochs)</span>
        <span><i className="swatch finetune" /> Fine-tuning ({finetuneValues.length} epochs)</span>
      </div>
    </figure>
  );
}
