import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The React plugin compiles JSX with the automatic runtime, so components do not
// need `import React`. Without this file Vite falls back to the classic
// React.createElement transform and the built dashboard fails at load time with
// "ReferenceError: React is not defined".
export default defineConfig({
  plugins: [react()],
});
