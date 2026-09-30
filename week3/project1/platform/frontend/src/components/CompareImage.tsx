import { useState } from "react";

/** Photo on the left, model overlay on the right; drag the divider to compare. */
export function CompareImage({ photo, overlay, alt, start = 50 }: { photo: string | null; overlay: string | null; alt: string; start?: number }) {
  const [pos, setPos] = useState(start);
  if (!photo) return <div className="compare compare-empty">No photo</div>;
  return (
    <div className="compare">
      <img src={overlay ?? photo} alt={`${alt}, model overlay`} draggable={false} />
      <div className="compare-photo" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <img src={photo} alt={alt} draggable={false} />
      </div>
      <div className="compare-handle" style={{ left: `clamp(14px, ${pos}%, calc(100% - 14px))` }} aria-hidden>
        <span />
      </div>
      <input
        className="compare-range"
        type="range"
        min={0}
        max={100}
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        aria-label="Slide between photo and overlay"
      />
      <span className="compare-label cl-left">Photo</span>
      <span className="compare-label cl-right">Overlay</span>
    </div>
  );
}
