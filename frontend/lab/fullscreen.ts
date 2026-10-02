import { useEffect, useRef } from "react";

/** Try to go fullscreen (needs a click). Fullscreen is optional: calibration is stored per screen size, so
 *  it only matters that runs happen the same way each time. Returns whether we are now fullscreen. */
export async function enterFullscreen(): Promise<boolean> {
  try {
    if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
    return !!document.fullscreenElement;
  } catch {
    return false;
  }
}

export function leaveFullscreen(): void {
  if (document.fullscreenElement) void document.exitFullscreen().catch(() => {});
}

/** While `active`, leaving fullscreen (Esc) or pressing Escape calls onExit. */
export function useExitGuard(active: boolean, onExit: () => void) {
  const cb = useRef(onExit);
  cb.current = onExit;
  useEffect(() => {
    if (!active) return;
    let wasFullscreen = !!document.fullscreenElement;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") cb.current(); };
    const onFs = () => {
      if (document.fullscreenElement) wasFullscreen = true;
      else if (wasFullscreen) cb.current();
    };
    window.addEventListener("keydown", onKey);
    document.addEventListener("fullscreenchange", onFs);
    return () => { window.removeEventListener("keydown", onKey); document.removeEventListener("fullscreenchange", onFs); };
  }, [active]);
}
