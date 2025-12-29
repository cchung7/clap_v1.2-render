/*
Vite configuration for CLAP frontend
- Local development: Uses proxy to localhost:5001 (Flask) when running standalone
- Vercel dev: API calls go through Vercel proxy to serverless functions
- Production: Uses same-domain API endpoints (serverless functions)
*/
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  },
  server: {
    // Only use Flask proxy when running standalone (not in Vercel)
    // In Vercel dev, API calls automatically go through Vercel's proxy
    proxy: (process.env.NODE_ENV === 'development' && !process.env.VERCEL && !process.env.VERCEL_ENV) ? {
      '/api': {
        target: "http://localhost:5001",
        changeOrigin: true,
      },
    } : undefined,
    port: process.env.PORT ? parseInt(process.env.PORT) : 5173,
    host: true,
  },
})
