import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        "bg-base":  "#0a0e1a",
        "bg-card":  "#111827",
        "bg-card2": "#1a2235",
      },
    },
  },
  plugins: [],
};

export default config;
