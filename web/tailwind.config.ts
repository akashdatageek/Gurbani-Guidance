import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        gurmukhi: ["Noto Sans Gurmukhi", "sans-serif"],
      },
      colors: {
        saffron: {
          50: "#fff9f0",
          100: "#fef3dc",
          200: "#fde3b0",
          300: "#fbc97a",
          400: "#f9a83d",
          500: "#f78b1a",
          600: "#e06b0e",
          700: "#b94f0e",
          800: "#943e13",
          900: "#783413",
        },
        navy: {
          800: "#1a2744",
          900: "#0f1a30",
        },
      },
    },
  },
  plugins: [],
};

export default config;
