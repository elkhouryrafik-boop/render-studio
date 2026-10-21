import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build lands in web/dist, which api/main.py serves directly. That is why
// there is no Node process at runtime — only at build time.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Source maps stay off: this is served on a LAN to colleagues, and the
    // bundle is smaller without them.
    sourcemap: false,
  },
  server: {
    port: 5173,
    // During development the API runs separately. Proxying keeps the browser
    // on one origin, so the session cookie behaves exactly as it will in
    // production instead of needing CORS rules that only exist in dev.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: false,
      },
    },
  },
});
