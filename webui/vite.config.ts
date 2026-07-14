import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../myclaw/web/dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8765',
    },
  },
  test: {
    environment: 'happy-dom',
    setupFiles: './src/tests/setup.ts',
    restoreMocks: true,
  },
})
