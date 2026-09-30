import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the UI on :5173 and forwards API + photo requests to the
// FastAPI backend on :8000 (start it with `python -m backend` in platform/).
const backend = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": backend,
      "/media": backend,
    },
  },
  build: { outDir: "dist", chunkSizeWarningLimit: 900 },
});
