import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        fg: "var(--fg)",
        muted: "var(--muted)",
        rule: "var(--rule)",
        soft: "var(--soft)",
        warn: "var(--warn)",
        danger: "var(--danger)",
      },
      fontFamily: {
        mono: "var(--font-mono)",
      },
      borderRadius: {
        none: "0px",
      },
    },
  },
  plugins: [],
};

export default config;
