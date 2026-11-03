export default function StatCard({ label, value, hint, tone }) {
  return (
    <div className={"stat-card" + (tone ? ` tone-${tone}` : "")}>
      <small>{label}</small>
      <b>{value}</b>
      {hint && <span>{hint}</span>}
    </div>
  );
}
