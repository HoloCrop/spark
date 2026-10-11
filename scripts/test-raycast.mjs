import path from "node:path";
import { pathToFileURL } from "node:url";
import { build } from "vite";
import arraybuffer from "vite-plugin-arraybuffer";
import glsl from "vite-plugin-glsl";

const outDir = "node_modules/.cache/spark-raycast-test";
await build({
  configFile: false,
  plugins: [arraybuffer(), glsl()],
  build: {
    target: "esnext",
    minify: false,
    outDir,
    lib: { entry: process.argv[2] ?? "test/raycast.test.ts", formats: ["es"], fileName: () => "raycast.mjs" },
    rollupOptions: { external: ["three", "node:assert/strict", "node:test"] },
  },
});
await import(pathToFileURL(path.resolve(outDir, "raycast.mjs")).href);
