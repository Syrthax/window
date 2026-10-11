import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// dev: the page runs on :5173 and proxies the API on :8000
const api = "http://127.0.0.1:8000";
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/today": api, "/eval": api, "/health": api } },
});
