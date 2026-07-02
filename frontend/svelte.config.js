import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default {
  preprocess: vitePreprocess(),

  kit: {
    adapter: adapter({
      pages: '../pynvr/frontend_dist',
      assets: '../pynvr/frontend_dist',
      fallback: 'index.html'
    })
  }
};
