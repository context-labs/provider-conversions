import { describe, it, expect, beforeEach } from "vitest";
import {
  googleChunkFromAiSDK,
  createChunkConversionContext,
  type ChunkConversionContext,
} from "~/google/genai/chunk-from-ai";
import type { AiSDKChunk } from "~/ai";
import type { ToolSet } from "ai";

describe("googleChunkFromAiSDK", () => {
  let context: ChunkConversionContext;

  beforeEach(() => {
    context = createChunkConversionContext({
      responseId: "resp_123abc",
      modelVersion: "gemini-2.0-flash",
    });
  });

  describe("createChunkConversionContext", () => {
    it("creates context with provided values", () => {
      const ctx = createChunkConversionContext({
        responseId: "test-id",
        modelVersion: "test-model",
      });

      expect(ctx.responseId).toBe("test-id");
      expect(ctx.modelVersion).toBe("test-model");
      expect(ctx.accumulatedText).toBe("");
      expect(ctx.accumulatedFunctionCalls.size).toBe(0);
      expect(ctx.inputTokens).toBe(0);
      expect(ctx.outputTokens).toBe(0);
    });
  });

  describe("text streaming", () => {
    it("skips text-start", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "text-start",
        id: "text-1",
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("converts text-delta to response with accumulated text", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "text-delta",
        id: "text-1",
        text: "Hello",
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("response");
      if (result.type === "response") {
        expect(result.response.responseId).toBe("resp_123abc");
        expect(result.response.modelVersion).toBe("gemini-2.0-flash");
        expect(result.response.candidates![0].content?.parts![0]).toEqual({
          text: "Hello",
        });
      }
    });

    it("accumulates text across multiple deltas", () => {
      const chunk1: AiSDKChunk<ToolSet> = {
        type: "text-delta",
        id: "text-1",
        text: "Hello",
      };
      const chunk2: AiSDKChunk<ToolSet> = {
        type: "text-delta",
        id: "text-1",
        text: ", ",
      };
      const chunk3: AiSDKChunk<ToolSet> = {
        type: "text-delta",
        id: "text-1",
        text: "world!",
      };

      googleChunkFromAiSDK(chunk1, context);
      googleChunkFromAiSDK(chunk2, context);
      const result = googleChunkFromAiSDK(chunk3, context);

      expect(result.type).toBe("response");
      if (result.type === "response") {
        expect(result.response.candidates![0].content?.parts![0]).toEqual({
          text: "Hello, world!",
        });
      }
    });

    it("skips text-end", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "text-end",
        id: "text-1",
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });
  });

  describe("reasoning streaming", () => {
    it("skips reasoning-start", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "reasoning-start",
        id: "reasoning-1",
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips reasoning-delta", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "reasoning-delta",
        id: "reasoning-1",
        text: "Let me think...",
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips reasoning-end", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "reasoning-end",
        id: "reasoning-1",
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });
  });

  describe("tool call streaming", () => {
    it("skips tool-input-start", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-start",
        id: "call_abc123",
        toolName: "get_weather",
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
      expect(context.accumulatedFunctionCalls.has("call_abc123")).toBe(true);
    });

    it("converts tool-input-delta to response", () => {
      // First, start the tool call
      const startChunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-start",
        id: "call_abc123",
        toolName: "get_weather",
      };
      googleChunkFromAiSDK(startChunk, context);

      // Then send delta
      const deltaChunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-delta",
        id: "call_abc123",
        delta: '{"location":',
      };

      const result = googleChunkFromAiSDK(deltaChunk, context);

      expect(result.type).toBe("response");
    });

    it("returns error for tool-input-delta with unknown id", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-delta",
        id: "unknown_call",
        delta: "{}",
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("error");
      if (result.type === "error") {
        expect(result.error).toContain("Unknown tool call id");
      }
    });

    it("converts tool-input-end to response with parsed args", () => {
      // Start the tool call
      const startChunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-start",
        id: "call_abc123",
        toolName: "get_weather",
      };
      googleChunkFromAiSDK(startChunk, context);

      // Send deltas
      const delta1: AiSDKChunk<ToolSet> = {
        type: "tool-input-delta",
        id: "call_abc123",
        delta: '{"location"',
      };
      const delta2: AiSDKChunk<ToolSet> = {
        type: "tool-input-delta",
        id: "call_abc123",
        delta: ':"SF"}',
      };
      googleChunkFromAiSDK(delta1, context);
      googleChunkFromAiSDK(delta2, context);

      // End the tool call
      const endChunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-end",
        id: "call_abc123",
      };

      const result = googleChunkFromAiSDK(endChunk, context);

      expect(result.type).toBe("response");
      if (result.type === "response") {
        const funcCall = result.response.candidates![0].content?.parts![0];
        expect(funcCall).toEqual({
          functionCall: {
            id: "call_abc123",
            name: "get_weather",
            args: { location: "SF" },
          },
        });
      }
    });

    it("converts complete tool-call chunk", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-call",
        toolCallId: "call_complete",
        toolName: "search",
        input: { query: "test" },
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("response");
      if (result.type === "response") {
        expect(result.response.candidates![0].content?.parts![0]).toEqual({
          functionCall: {
            id: "call_complete",
            name: "search",
            args: { query: "test" },
          },
        });
      }
    });

    it("skips tool-result", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-result",
        toolCallId: "call_123",
        toolName: "get_weather",
        output: { type: "text", value: "Sunny" },
        input: {},
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips tool-error", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-error",
        toolCallId: "call_123",
        toolName: "failing_tool",
        input: {},
        error: new Error("Tool failed"),
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });
  });

  describe("finish events", () => {
    it("skips finish-step but tracks usage", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "finish-step",
        response: {
          id: "resp-1",
          timestamp: new Date(),
          modelId: "gemini-2.0-flash",
        },
        usage: {
          inputTokens: 10,
          outputTokens: 20,
          totalTokens: 30,
        },
        finishReason: "stop",
        providerMetadata: undefined,
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
      expect(context.inputTokens).toBe(10);
      expect(context.outputTokens).toBe(20);
    });

    it("converts finish to final response with stop reason", () => {
      // First add some text
      const textChunk: AiSDKChunk<ToolSet> = {
        type: "text-delta",
        id: "text-1",
        text: "Final answer",
      };
      googleChunkFromAiSDK(textChunk, context);

      // Then finish
      const finishChunk: AiSDKChunk<ToolSet> = {
        type: "finish",
        finishReason: "stop",
        totalUsage: {
          inputTokens: 100,
          outputTokens: 50,
          totalTokens: 150,
        },
      };

      const result = googleChunkFromAiSDK(finishChunk, context);

      expect(result.type).toBe("response");
      if (result.type === "response") {
        expect(result.response.candidates![0].finishReason).toBe("STOP");
        expect(result.response.usageMetadata?.promptTokenCount).toBe(100);
        expect(result.response.usageMetadata?.candidatesTokenCount).toBe(50);
        expect(result.response.usageMetadata?.totalTokenCount).toBe(150);
      }
    });

    it("converts length finish reason to MAX_TOKENS", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "finish",
        finishReason: "length",
        totalUsage: {
          inputTokens: 100,
          outputTokens: 4096,
          totalTokens: 4196,
        },
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("response");
      if (result.type === "response") {
        expect(result.response.candidates![0].finishReason).toBe("MAX_TOKENS");
      }
    });

    it("converts content-filter finish reason to SAFETY", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "finish",
        finishReason: "content-filter",
        totalUsage: {
          inputTokens: 10,
          outputTokens: 0,
          totalTokens: 10,
        },
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("response");
      if (result.type === "response") {
        expect(result.response.candidates![0].finishReason).toBe("SAFETY");
      }
    });

    it("converts tool-calls finish reason to STOP", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "finish",
        finishReason: "tool-calls",
        totalUsage: {
          inputTokens: 50,
          outputTokens: 25,
          totalTokens: 75,
        },
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("response");
      if (result.type === "response") {
        expect(result.response.candidates![0].finishReason).toBe("STOP");
      }
    });
  });

  describe("lifecycle events", () => {
    it("skips start", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "start",
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips start-step", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "start-step",
        request: {},
        warnings: [],
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips abort", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "abort",
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });
  });

  describe("error handling", () => {
    it("converts error chunk to error result", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "error",
        error: new Error("Something went wrong"),
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("error");
      if (result.type === "error") {
        expect(result.error).toContain("Something went wrong");
      }
    });

    it("handles non-Error error values", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "error",
        error: "String error message",
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("error");
      if (result.type === "error") {
        expect(result.error).toBe("String error message");
      }
    });
  });

  describe("other chunk types", () => {
    it("skips source chunks", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "source",
        id: "source-1",
        sourceType: "url",
        url: "https://example.com",
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips file chunks", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "file",
        file: {
          base64: "data...",
          uint8Array: new Uint8Array(),
          mediaType: "image/png",
        },
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips raw chunks", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "raw",
        rawValue: { some: "data" },
      };

      const result = googleChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });
  });

  describe("full streaming sequence", () => {
    it("handles a complete text streaming sequence", () => {
      const chunks: AiSDKChunk<ToolSet>[] = [
        { type: "start" },
        { type: "start-step", request: {}, warnings: [] },
        { type: "text-start", id: "text-1" },
        { type: "text-delta", id: "text-1", text: "Hello" },
        { type: "text-delta", id: "text-1", text: ", " },
        { type: "text-delta", id: "text-1", text: "world!" },
        { type: "text-end", id: "text-1" },
        {
          type: "finish-step",
          response: { id: "r1", timestamp: new Date(), modelId: "gemini-2.0-flash" },
          usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
          finishReason: "stop",
          providerMetadata: undefined,
        },
        {
          type: "finish",
          finishReason: "stop",
          totalUsage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
        },
      ];

      const responses: Array<{ candidates: unknown }> = [];
      for (const chunk of chunks) {
        const result = googleChunkFromAiSDK(chunk, context);
        if (result.type === "response") {
          responses.push(result.response);
        }
      }

      // Should have 4 responses: 3 text-deltas + 1 finish
      expect(responses).toHaveLength(4);

      // Final response should have complete text
      const finalResponse = responses[responses.length - 1];
      expect((finalResponse.candidates as Array<{ content: { parts: Array<{ text: string }> } }>)[0].content.parts[0].text).toBe("Hello, world!");
    });

    it("handles a complete tool call streaming sequence", () => {
      const chunks: AiSDKChunk<ToolSet>[] = [
        { type: "start" },
        { type: "text-start", id: "text-1" },
        { type: "text-delta", id: "text-1", text: "Let me check" },
        { type: "text-end", id: "text-1" },
        { type: "tool-input-start", id: "call_1", toolName: "get_weather" },
        { type: "tool-input-delta", id: "call_1", delta: '{"location"' },
        { type: "tool-input-delta", id: "call_1", delta: ':"SF"}' },
        { type: "tool-input-end", id: "call_1" },
        {
          type: "finish",
          finishReason: "tool-calls",
          totalUsage: { inputTokens: 20, outputTokens: 15, totalTokens: 35 },
        },
      ];

      const responses: Array<{ candidates: unknown }> = [];
      for (const chunk of chunks) {
        const result = googleChunkFromAiSDK(chunk, context);
        if (result.type === "response") {
          responses.push(result.response);
        }
      }

      // Final response should have both text and function call
      const finalResponse = responses[responses.length - 1];
      const parts = (finalResponse.candidates as Array<{ content: { parts: Array<{ text?: string; functionCall?: unknown }> } }>)[0].content.parts;

      expect(parts).toHaveLength(2);
      expect(parts[0].text).toBe("Let me check");
      expect(parts[1].functionCall).toEqual({
        id: "call_1",
        name: "get_weather",
        args: { location: "SF" },
      });
    });
  });
});
