import { useEffect } from "react";
import Home from "./components/Home";
import CaseView from "./components/CaseView";
import Toast from "./components/Toast";
import { useStore } from "./lib/store";

// Minimal hash routing: #/ (home) and #/case/<id>
function parseHash(): string | null {
  const m = window.location.hash.match(/^#\/case\/([0-9a-f]+)/);
  return m ? m[1] : null;
}

export default function App() {
  const caseId = useStore((s) => s.caseId);
  const setCase = useStore((s) => s.setCase);
  useEffect(() => {
    const onHash = () => setCase(parseHash());
    onHash();
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [setCase]);
  return (
    <>
      {caseId ? <CaseView caseId={caseId} /> : <Home />}
      <Toast />
    </>
  );
}

export const openCase = (id: string) => (window.location.hash = `#/case/${id}`);
export const goHome = () => (window.location.hash = "#/");
