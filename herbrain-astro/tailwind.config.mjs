/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        'herbrain-green': '#4A7C6F',
        'herbrain-coral': '#E8927C',
        'herbrain-bg': '#FAFBFC',
        'herbrain-card': '#FFFFFF',
        'herbrain-dark': '#1A1A2E',
        'herbrain-muted': '#6B7280',
      },
      fontFamily: {
        'display': ['"Playfair Display"', 'Georgia', 'serif'],
        'sans': ['Inter', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
