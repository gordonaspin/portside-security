import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default {
  preprocess: vitePreprocess(),

  kit: {
    adapter: adapter({
      pages: '../backend/frontend_dist',
      assets: '../backend/frontend_dist',
      fallback: 'index.html'
    })
  }
};
