/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#4BFF4B",
        secondary: "#33FF33",
        nprimary: "#FF4B4B",
        nprimary20: "#FF4B4B33",
        nsecondary: "#FF3333",
      },
    },
  },
  plugins: [],
}