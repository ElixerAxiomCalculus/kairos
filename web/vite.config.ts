import { fileURLToPath, URL } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In dev, /api and /ws are proxied to the Kairos backend (python -m kairos serve).
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
  build: {
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: {
          d3: ["d3"],
          recharts: ["recharts"],
          motion: ["framer-motion"],
          radix: [
            "@radix-ui/react-dialog",
            "@radix-ui/react-popover",
            "@radix-ui/react-tabs",
            "@radix-ui/react-toggle-group",
            "@radix-ui/react-tooltip",
          ],
        },
      },
    },
  },
});
