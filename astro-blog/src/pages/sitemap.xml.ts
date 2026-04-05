import { getCollection } from 'astro:content';
import type { APIRoute } from 'astro';

const SITE = 'https://xkhunx.github.io';

function cleanSlug(s: string) {
  return s.replace(/^\d{4}-\d{2}-\d{2}-/, '');
}

function postUrl(post: Awaited<ReturnType<typeof getCollection<'posts'>>>[0]) {
  const d = post.data.date;
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${SITE}/${y}/${m}/${day}/${cleanSlug(post.slug)}.html`;
}

export const GET: APIRoute = async () => {
  const posts = (await getCollection('posts')).sort(
    (a, b) => b.data.date.valueOf() - a.data.date.valueOf()
  );

  const pages = [
    { url: `${SITE}/`, priority: '1.0' },
    { url: `${SITE}/tags.html`, priority: '0.8' },
    ...posts.map(p => ({ url: postUrl(p), priority: '0.7' })),
  ];

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${pages.map(({ url, priority }) =>
  `  <url>\n    <loc>${url}</loc>\n    <priority>${priority}</priority>\n  </url>`
).join('\n')}
</urlset>`;

  return new Response(xml, {
    headers: { 'Content-Type': 'application/xml; charset=utf-8' },
  });
};
