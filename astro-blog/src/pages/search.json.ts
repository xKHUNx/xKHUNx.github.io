import { getCollection } from 'astro:content';
import type { APIRoute } from 'astro';

export const GET: APIRoute = async () => {
  const posts = (await getCollection('posts')).sort(
    (a, b) => b.data.date.valueOf() - a.data.date.valueOf()
  );

  const searchIndex = posts.map(post => {
    const d = post.data.date;
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return {
      title: post.data.title,
      tags: post.data.tags.join(' '),
      url: `/${y}/${m}/${day}/${post.slug.replace(/^\d{4}-\d{2}-\d{2}-/, '')}`,
    };
  });

  return new Response(JSON.stringify(searchIndex), {
    headers: { 'Content-Type': 'application/json' },
  });
};
