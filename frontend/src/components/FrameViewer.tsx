import { useEffect, useRef, useState } from "react";
import { frameUrl } from "../lib/api";
import { useStore } from "../lib/store";

// Exact source frame decoded from the evidence file (not the review proxy).
export default function FrameViewer({ caseId }: { caseId: string }) {
  const fv = useStore((s) => s.frameViewer);
  const close = () => useStore.getState().openFrame(null);
  const img = useRef<HTMLImageElement>(null);
  const [size, setSize] = useState<[number, number] | null>(null);
  useEffect(() => setSize(null), [fv?.video, fv?.frame]);
  if (!fv) return null;
  const url = frameUrl(caseId, fv.video, fv.frame);
  return (
    <div className="overlay" onClick={close}>
      <div className="modal wide" onClick={(e) => e.stopPropagation()}>
        <div className="row">
          <h3 className="grow">{fv.title ?? "Source frame"} — {fv.video} frame {fv.frame}</h3>
          <button className="icon" onClick={() => useStore.getState().openFrame({ ...fv, frame: Math.max(0, fv.frame - 1) })}>◀</button>
          <button className="icon" onClick={() => useStore.getState().openFrame({ ...fv, frame: fv.frame + 1 })}>▶</button>
          <a href={url} download={`${fv.video}_frame_${fv.frame}.png`}>download PNG</a>
          <button className="icon" onClick={close}>✕</button>
        </div>
        <div className="muted small" style={{ margin: "4px 0 8px" }}>Decoded directly from the original evidence file (lossless PNG of the decoded frame). Box: detector output for this observation.</div>
        <div style={{ position: "relative", display: "inline-block", maxWidth: "100%" }}>
          <img ref={img} src={url} style={{ maxWidth: "100%", display: "block" }} onLoad={(e) => setSize([e.currentTarget.naturalWidth, e.currentTarget.clientWidth])} />
          {fv.bbox && size && (() => {
            const s = size[1] / size[0];
            const [x1, y1, x2, y2] = fv.bbox;
            return <div style={{ position: "absolute", left: x1 * s, top: y1 * s, width: (x2 - x1) * s, height: (y2 - y1) * s, border: "2px solid #ffb347", pointerEvents: "none" }} />;
          })()}
        </div>
      </div>
    </div>
  );
}
