import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
export default {
  preprocess: vitePreprocess(),
  
  // Svelte 4's native hook to intercept and silence compiler warnings
  onwarn: (warning, handler) => {
    if (warning.code === 'a11y-click-events-have-key-events') return;
    if (warning.code === 'a11y-label-has-associated-control') return;
    if (warning.code === 'a11y-no-static-element-interactions') return;
    handler(warning); // Let all other warnings pass through to the console
  },

  kit: {
    adapter: adapter({
      pages: '../pynvr/frontend_dist',
      assets: '../pynvr/frontend_dist',
      fallback: 'index.html'
    })
  }
};
