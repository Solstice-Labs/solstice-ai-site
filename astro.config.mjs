import { defineConfig } from 'astro/config';
import cloudflare from '@astrojs/cloudflare';

export default defineConfig({
  site: 'https://solstice-ai.co',
  output: 'static',
  adapter: cloudflare(),
});
