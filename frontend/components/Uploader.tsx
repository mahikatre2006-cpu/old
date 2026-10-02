import { useEffect, useRef, useState } from "react";

const MAX_BYTES = 10 * 1024 * 1024;
const OK_TYPES = ["image/jpeg", "image/png", "image/webp"];

interface Props {
  busy: boolean;
  onFile: (file: File) => void;
  onError: (msg: string) => void;
  compact?: boolean;
}

export default function Uploader({ busy, onFile, onError, compact }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [samples, setSamples] = useState<{ file: string; label: string }[]>([]);

  function accept(file: File | undefined) {
    if (!file) return;
    if (!OK_TYPES.includes(file.type)) return onError("Please use a JPG, PNG or WEBP image.");
    if (file.size > MAX_BYTES) return onError("That file is over 10 MB.");
    onFile(file);
  }

  // Ctrl+V anywhere on the page
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => accept([...(e.clipboardData?.files ?? [])][0]);
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  });

  // Dev convenience: frontend/public/samples/index.json (git-ignored) lists local sample thumbnails.
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    fetch("/samples/index.json").then((r) => (r.ok ? r.json() : [])).then(setSamples).catch(() => {});
  }, []);

  async function useSample(file: string) {
    const blob = await (await fetch(`/samples/${file}`)).blob();
    accept(new File([blob], file, { type: blob.type || "image/jpeg" }));
  }

  return (
    <div className="w-full">
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); accept(e.dataTransfer.files[0]); }}
        className={`flex flex-col items-center justify-center rounded-2xl border-2 border-dashed text-center transition-colors
          ${compact ? "px-4 py-5" : "px-6 py-16"}
          ${drag ? "border-amber-500 bg-amber-50 dark:bg-amber-950/30" : "border-stone-300 dark:border-stone-700"}`}
      >
        <p className={`${compact ? "text-sm" : "text-lg"} font-medium`}>
          {busy ? "Analysing…" : compact ? "Analyse another thumbnail" : "Drop a thumbnail here"}
        </p>
        <p className="mt-1 text-sm text-stone-500 dark:text-stone-400">JPG, PNG or WEBP, up to 10 MB. You can also paste with Ctrl+V.</p>
        <button
          type="button"
          disabled={busy}
          onClick={() => input.current?.click()}
          className="mt-4 rounded-lg bg-stone-900 px-4 py-2 text-sm font-medium text-white hover:bg-stone-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-500 disabled:opacity-50 dark:bg-amber-400 dark:text-stone-950 dark:hover:bg-amber-300"
        >
          Choose a file
        </button>
        <input ref={input} type="file" accept={OK_TYPES.join(",")} hidden onChange={(e) => { accept(e.target.files?.[0]); e.target.value = ""; }} />
      </div>

      {samples.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
          <span className="text-stone-500 dark:text-stone-400">Dev samples:</span>
          {samples.map((s) => (
            <button key={s.file} disabled={busy} onClick={() => useSample(s.file)} className="rounded-md border border-stone-300 px-2 py-1 hover:bg-stone-100 disabled:opacity-50 dark:border-stone-700 dark:hover:bg-stone-800">
              {s.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
