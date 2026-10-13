import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs on :8765 (metrica serve --no-browser); Vite proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 2000 },
});
