import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
  const posts = (await getCollection('posts')).sort(
    (a, b) => b.data.date.valueOf() - a.data.date.valueOf()
  );

  return rss({
    title: "Khun's Blog",
    description: '昆虫的部落客',
    site: context.site!,
    items: posts.map(post => {
      const d = post.data.date;
      const y = d.getFullYear();
      const m = String(d.getMonth() + 1).padStart(2, '0');
      const day = String(d.getDate()).padStart(2, '0');
      return {
        title: post.data.title,
        pubDate: post.data.date,
        link: `/${y}/${m}/${day}/${post.slug.replace(/^\d{4}-\d{2}-\d{2}-/, '')}`,
      };
    }),
  });
}
