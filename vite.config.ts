import { defineConfig } from "vite";
import dts from "vite-plugin-dts";
import { resolve } from "path";

export default defineConfig({
  plugins: [
    dts({
      include: ["src"],
      rollupTypes: false,
    }),
  ],
  resolve: {
    alias: {
      "~": resolve(__dirname, "src"),
    },
  },
  build: {
    lib: {
      entry: resolve(__dirname, "src/index.ts"),
      name: "InferenceNetProviderConversions",
      fileName: "inference-net-provider-conversions",
      formats: ["es", "cjs"],
    },
    rollupOptions: {
      external: [
        "openai",
        "@anthropic-ai/sdk",
        "@google/genai",
        "ai",
        "@ai-sdk/provider",
      ],
    },
  },
});
