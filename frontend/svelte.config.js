import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
export default {
  preprocess: vitePreprocess(),
  compilerOptions: {
    // Silences warnings during build and development compilations
    warningFilter: (warning) => warning.code !== 'a11y-click-events-have-key-events'
  },
  kit: {
    adapter: adapter({
      pages: '../pynvr/frontend_dist',
      assets: '../pynvr/frontend_dist',
      fallback: 'index.html'
    })
  }
};
