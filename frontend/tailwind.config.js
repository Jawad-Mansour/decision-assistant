/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        skin: {
          bg: "#F7F7F8",
          card: "#FFFFFF",
          border: "#E5E7EB",
          text: "#111827",
          muted: "#6B7280",
          primary: "#4F46E5",
          accent: "#7C3AED",
          urgent: "#DC2626",
          normal: "#10B981",
        },
      },
      boxShadow: {
        card: "0 1px 2px rgba(17, 24, 39, 0.04), 0 4px 16px rgba(17, 24, 39, 0.05)",
        glow: "0 10px 30px -12px rgba(79, 70, 229, 0.35)",
      },
      keyframes: {
        "fade-in-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "logo-float": {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-6px)" },
        },
        "ring-pulse": {
          "0%": { boxShadow: "0 0 0 0 rgba(79, 70, 229, 0.35)" },
          "70%": { boxShadow: "0 0 0 14px rgba(79, 70, 229, 0)" },
          "100%": { boxShadow: "0 0 0 0 rgba(79, 70, 229, 0)" },
        },
        "heart-pulse": {
          "0%, 100%": { transform: "scale(1)" },
          "30%": { transform: "scale(1.18)" },
          "55%": { transform: "scale(0.95)" },
          "75%": { transform: "scale(1.08)" },
        },
      },
      animation: {
        "fade-in-up": "fade-in-up 300ms ease-out both",
        "fade-in": "fade-in 300ms ease-out both",
        "logo-float": "logo-float 4s ease-in-out infinite",
        "ring-pulse": "ring-pulse 2.2s ease-out infinite",
        "heart-pulse": "heart-pulse 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
