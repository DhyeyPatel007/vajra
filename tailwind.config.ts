import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0B0C0E",
        surface: "#121418",
        panel: "#181B22",
        border: "#262A34",
        gold: {
          DEFAULT: "#D4AF37",
          bright: "#F5C542",
          dim: "#8A7322",
        },
        charcoal: {
          950: "#07080A",
          900: "#0B0C0E",
          850: "#121418",
          800: "#181B22",
          700: "#262A34",
          600: "#383E4D",
        },
        verified: "#10B981",
        failed: "#EF4444",
      },
    },
  },
  plugins: [],
};

export default config;
