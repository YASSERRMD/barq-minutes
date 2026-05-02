import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';
import { resolve } from 'node:path';

export default defineConfig({
  plugins: [wasm(), topLevelAwait(), react()],
  resolve: {
    alias: {
      'barq-vweb': resolve(__dirname, 'public/barq-vweb-pkg'),
      'barq-wasm': resolve(__dirname, 'public/barq-wasm-pkg'),
    },
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    exclude: ['@huggingface/transformers', 'onnxruntime-web', 'barq-vweb', 'barq-wasm'],
  },
});
