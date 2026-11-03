import { displayClassName } from "../lib/format.js";

/**
 * Confusion matrix as a real HTML table (not just an image) so screen
 * readers and text-only review get the exact counts. Rows = true label,
 * columns = predicted label -- stated explicitly, not left implicit.
 */
export default function ConfusionMatrixTable({ matrix, classNames }) {
  const max = Math.max(...matrix.flat());

  return (
    <div className="confusion-wrap">
      <table className="confusion-table" aria-label="Confusion matrix: rows are the true label, columns are the predicted label">
        <caption>
          Confusion matrix — rows: true label, columns: predicted label
        </caption>
        <thead>
          <tr>
            <th scope="col" className="confusion-corner">
              True \ Predicted
            </th>
            {classNames.map((name) => (
              <th scope="col" key={name}>
                {displayClassName(name)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {classNames.map((rowName, rowIndex) => (
            <tr key={rowName}>
              <th scope="row">{displayClassName(rowName)}</th>
              {classNames.map((colName, colIndex) => {
                const value = matrix[rowIndex][colIndex];
                const intensity = max ? value / max : 0;
                const isDiagonal = rowIndex === colIndex;
                return (
                  <td
                    key={colName}
                    className={isDiagonal ? "confusion-diag" : undefined}
                    style={{
                      background: isDiagonal
                        ? `rgba(20, 108, 112, ${0.12 + intensity * 0.55})`
                        : value > 0
                          ? `rgba(166, 58, 58, ${0.1 + intensity * 0.45})`
                          : "transparent",
                    }}
                  >
                    {value}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
