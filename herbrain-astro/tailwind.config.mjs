/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        'herbrain-green': '#3D7A6B',
        'herbrain-green-light': '#4A9385',
        'herbrain-green-dark': '#2D5A4F',
        'herbrain-coral': '#E8927C',
        'herbrain-bg': '#F8FAFB',
        'herbrain-card': '#FFFFFF',
        'herbrain-dark': '#0F172A',
        'herbrain-muted': '#64748B',
        'herbrain-border': '#E2E8F0',
        'herbrain-surface': '#F1F5F9',
      },
      fontFamily: {
        'display': ['"Playfair Display"', 'Georgia', 'serif'],
        'sans': ['Inter', '-apple-system', 'sans-serif'],
      },
      boxShadow: {
        'premium': '0 1px 3px 0 rgb(0 0 0 / 0.04), 0 1px 2px -1px rgb(0 0 0 / 0.04)',
        'premium-md': '0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05)',
        'premium-lg': '0 10px 15px -3px rgb(0 0 0 / 0.05), 0 4px 6px -4px rgb(0 0 0 / 0.05)',
        'glow': '0 0 20px -5px rgb(61 122 107 / 0.15)',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'glass': 'linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%)',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'pulse-subtle': 'pulseSubtle 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseSubtle: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.85' },
        },
      },
    },
  },
  plugins: [],
};
