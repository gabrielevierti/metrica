import { useStore } from "../lib/store";

export default function Toast() {
  const t = useStore((s) => s.toast);
  if (!t) return null;
  return <div className={`toast ${t.kind}`}>{t.text}</div>;
}
