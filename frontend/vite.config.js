import { sveltekit } from '@sveltejs/kit/vite';

export default {
  build: {
    sourcemap: true
  },
  plugins: [sveltekit()],
  server: {
    proxy: {
      '/api': 'http://localhost:7860',
      '/signal': 'http://localhost:7860'
    }
  }
};