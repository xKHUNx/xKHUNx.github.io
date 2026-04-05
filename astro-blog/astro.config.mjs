import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

export default defineConfig({
  site: 'https://xkhunx.github.io',
  integrations: [mdx()],
  build: {
    // Generates /2020/11/22/who-is-chan-sow-lin.html matching Jekyll's permalink format
    format: 'file',
  },
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
    },
  },
});
