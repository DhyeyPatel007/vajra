import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)", "Space Grotesk", "sans-serif"],
        mono: ["var(--font-mono)", "JetBrains Mono", "monospace"],
      },
      colors: {
        canvas: "#07080A",
        surface: "#0E1014",
        well: "#0A0C0F",
        elevated: "#151921",
        gridline: "#242936",
        gold: {
          DEFAULT: "#E5B842",
          bright: "#F5C542",
          dim: "#8A6D1F",
          dark: "#171308",
        },
        slate: {
          primary: "#F4F6F8",
          secondary: "#8E96A8",
          muted: "#5E6678",
        },
        verified: {
          DEFAULT: "#10B981",
          bg: "#09221B",
        },
        failed: {
          DEFAULT: "#F43F5E",
          bg: "#280B12",
        },
      },
    },
  },
  plugins: [],
};

export default config;
