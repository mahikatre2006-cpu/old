// WebGazer is a classic script plus MediaPipe face-mesh files (WASM + model, ~17 MB). We serve both from our own
// origin so the eye lab works offline. Runs automatically before dev and build.
import { copyFileSync, cpSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "node_modules", "webgazer", "dist");
const from = join(dist, "mediapipe");
const to = join(root, "public", "mediapipe");

if (!existsSync(from)) {
  console.warn("[sync-webgazer-assets] webgazer is not installed; the eye lab will not work. Run `npm install`.");
  process.exit(0);
}
mkdirSync(to, { recursive: true });
cpSync(from, to, { recursive: true });
copyFileSync(join(dist, "webgazer.js"), join(root, "public", "webgazer.js"));
console.log("[sync-webgazer-assets] copied webgazer.js and the face-mesh files into public/");
