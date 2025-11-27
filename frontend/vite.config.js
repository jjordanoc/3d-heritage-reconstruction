import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import wasm from 'vite-plugin-wasm'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [vue(), wasm()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) }
  },
  optimizeDeps: {
    exclude: ['@sparkjsdev/spark']
  },
  server: {
    host: '0.0.0.0',
    allowedHosts: true, // Allow all hostnames
    // ... other server options
  }
})
