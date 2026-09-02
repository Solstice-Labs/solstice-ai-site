import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const papers = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/papers' }),
  schema: z.object({
    title: z.string(),
    authors: z.array(z.string()),
    pubDate: z.coerce.date(),
    tldr: z.string(),
    abstract: z.string(),
    venue: z.string().default('Technical Report'),
    arxivId: z.string().optional(),
    arxivUrl: z.string().optional(),
    pdfUrl: z.string().optional(),
    huggingfaceUrl: z.string().optional(),
    githubUrl: z.string().optional(),
    highlightMetrics: z.array(z.object({
      label: z.string(),
      value: z.string(),
    })).optional(),
    bibtex: z.string(),
    tags: z.array(z.string()).default([]),
    featured: z.boolean().default(false),
  }),
});

const blog = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    author: z.string().default('Solstice-AI Research'),
    tags: z.array(z.string()).default([]),
    readingTime: z.string().optional(),
    takeaways: z.array(z.string()).optional(),
    featured: z.boolean().default(false),
  }),
});

const docs = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/docs' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    category: z.enum(['datasets', 'models', 'tooling', 'guides']),
    order: z.number().default(0),
    lastUpdated: z.coerce.date().optional(),
    hfRepoId: z.string().optional(),
    githubUrl: z.string().optional(),
    specs: z.record(z.string(), z.string()).optional(),
    supportedFormats: z.array(z.string()).optional(),
  }),
});

export const collections = { papers, blog, docs };
