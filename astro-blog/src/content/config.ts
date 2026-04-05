import { defineCollection, z } from 'astro:content';

const posts = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    author: z.string().default('Khun'),
    cover: z.string().optional(),
    thumbnail: z.string().optional(),
    tags: z.union([
      z.string().transform(s => s.split(' ').filter(Boolean)),
      z.array(z.string()),
    ]).default([]),
    subtitle: z.string().optional(),
  }),
});

export const collections = { posts };
