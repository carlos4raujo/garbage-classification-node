import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    mimeTypes: {
      "application/octet-stream": ["bin"],
    },
  },
  resolve: {
    alias: {
      assets: "src/assets",
    },
  },
})
