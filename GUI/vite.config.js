import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5000,
    host: '0.0.0.0',
    strictPort: true,
    hmr: {
      clientPort: 443,
      protocol: 'wss'
    },
    proxy: {
      '/api': {
        target: `http://localhost:${process.env.API_PORT || '8000'}`,
        changeOrigin: true,
      },
      '/metrics': {
        target: `http://localhost:${process.env.API_PORT || '8000'}`,
        changeOrigin: true,
      },
      '/ws': {
        target: `ws://localhost:${process.env.API_PORT || '8000'}`,
        ws: true,
        changeOrigin: true,
      },
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
}) 