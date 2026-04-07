import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

export default defineConfig({
  site: 'https://xkhunx.github.io',
  integrations: [mdx()],
  redirects: {
    '/2019/08/18/welcome-to-khuns-blog.html': '/2019/08/18/welcome-to-khuns-blog',
    '/2019/08/25/the-mystery-of-the-broken-bridge-in-putrajaya.html': '/2019/08/25/the-mystery-of-the-broken-bridge-in-putrajaya',
    '/2019/09/15/the-creature-with-magical-ballsack.html': '/2019/09/15/the-creature-with-magical-ballsack',
    '/2020/03/26/machine-learning-for-beginners-using-ai-to-write-simple-functions-like-iseven.html': '/2020/03/26/machine-learning-for-beginners-using-ai-to-write-simple-functions-like-iseven',
    '/2020/11/22/who-is-chan-sow-lin.html': '/2020/11/22/who-is-chan-sow-lin',
  },
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
    },
  },
});
