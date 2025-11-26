import { describe, it, expect, beforeEach } from "vitest";
import {
  openaiChunkFromAiSDK,
  createChunkConversionContext,
  type ChunkConversionContext,
} from "~/openai/chat-completion/chunk-from-ai";
import type { AiSDKChunk } from "~/ai";
import type { ToolSet } from "ai";

describe("openaiChunkFromAiSDK", () => {
  let context: ChunkConversionContext;

  beforeEach(() => {
    context = createChunkConversionContext({
      id: "chatcmpl-123",
      model: "gpt-4o",
      created: 1700000000,
    });
  });

  describe("createChunkConversionContext", () => {
    it("creates context with provided values", () => {
      const ctx = createChunkConversionContext({
        id: "test-id",
        model: "test-model",
        created: 1234567890,
      });

      expect(ctx.id).toBe("test-id");
      expect(ctx.model).toBe("test-model");
      expect(ctx.created).toBe(1234567890);
      expect(ctx.toolCallIndices.size).toBe(0);
      expect(ctx.nextToolCallIndex).toBe(0);
    });

    it("defaults created to current time if not provided", () => {
      const before = Math.floor(Date.now() / 1000);
      const ctx = createChunkConversionContext({
        id: "test-id",
        model: "test-model",
      });
      const after = Math.floor(Date.now() / 1000);

      expect(ctx.created).toBeGreaterThanOrEqual(before);
      expect(ctx.created).toBeLessThanOrEqual(after);
    });
  });

  describe("text streaming", () => {
    it("converts text-start to initial chunk with role", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "text-start",
        id: "text-1",
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("chunk");
      if (result.type === "chunk") {
        expect(result.chunk.id).toBe("chatcmpl-123");
        expect(result.chunk.object).toBe("chat.completion.chunk");
        expect(result.chunk.created).toBe(1700000000);
        expect(result.chunk.model).toBe("gpt-4o");
        expect(result.chunk.choices[0].delta).toEqual({
          role: "assistant",
          content: "",
        });
        expect(result.chunk.choices[0].finish_reason).toBeNull();
      }
    });

    it("converts text-delta to content chunk", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "text-delta",
        id: "text-1",
        text: "Hello",
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("chunk");
      if (result.type === "chunk") {
        expect(result.chunk.choices[0].delta).toEqual({ content: "Hello" });
        expect(result.chunk.choices[0].finish_reason).toBeNull();
      }
    });

    it("skips text-end", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "text-end",
        id: "text-1",
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });
  });

  describe("reasoning streaming", () => {
    it("skips reasoning-start", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "reasoning-start",
        id: "reasoning-1",
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips reasoning-delta", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "reasoning-delta",
        id: "reasoning-1",
        text: "Let me think...",
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips reasoning-end", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "reasoning-end",
        id: "reasoning-1",
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });
  });

  describe("tool call streaming", () => {
    it("converts tool-input-start to tool call chunk with id and name", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-start",
        id: "call_abc123",
        toolName: "get_weather",
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("chunk");
      if (result.type === "chunk") {
        expect(result.chunk.choices[0].delta.tool_calls).toEqual([
          {
            index: 0,
            id: "call_abc123",
            type: "function",
            function: {
              name: "get_weather",
              arguments: "",
            },
          },
        ]);
      }
    });

    it("converts tool-input-delta to arguments delta chunk", () => {
      // First, start the tool call
      const startChunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-start",
        id: "call_abc123",
        toolName: "get_weather",
      };
      openaiChunkFromAiSDK(startChunk, context);

      // Then send delta
      const deltaChunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-delta",
        id: "call_abc123",
        delta: '{"location":',
      };

      const result = openaiChunkFromAiSDK(deltaChunk, context);

      expect(result.type).toBe("chunk");
      if (result.type === "chunk") {
        expect(result.chunk.choices[0].delta.tool_calls).toEqual([
          {
            index: 0,
            function: {
              arguments: '{"location":',
            },
          },
        ]);
      }
    });

    it("returns error for tool-input-delta with unknown id", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-delta",
        id: "unknown_call",
        delta: "{}",
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("error");
      if (result.type === "error") {
        expect(result.error).toContain("Unknown tool call id");
      }
    });

    it("skips tool-input-end", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-end",
        id: "call_abc123",
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("assigns sequential indices to multiple tool calls", () => {
      const chunk1: AiSDKChunk<ToolSet> = {
        type: "tool-input-start",
        id: "call_1",
        toolName: "tool_a",
      };
      const chunk2: AiSDKChunk<ToolSet> = {
        type: "tool-input-start",
        id: "call_2",
        toolName: "tool_b",
      };

      const result1 = openaiChunkFromAiSDK(chunk1, context);
      const result2 = openaiChunkFromAiSDK(chunk2, context);

      expect(result1.type).toBe("chunk");
      expect(result2.type).toBe("chunk");
      if (result1.type === "chunk" && result2.type === "chunk") {
        expect(result1.chunk.choices[0].delta.tool_calls![0].index).toBe(0);
        expect(result2.chunk.choices[0].delta.tool_calls![0].index).toBe(1);
      }
    });

    it("reuses index for same tool call id", () => {
      const startChunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-start",
        id: "call_abc123",
        toolName: "get_weather",
      };
      const deltaChunk1: AiSDKChunk<ToolSet> = {
        type: "tool-input-delta",
        id: "call_abc123",
        delta: '{"loc',
      };
      const deltaChunk2: AiSDKChunk<ToolSet> = {
        type: "tool-input-delta",
        id: "call_abc123",
        delta: 'ation":"SF"}',
      };

      const result1 = openaiChunkFromAiSDK(startChunk, context);
      const result2 = openaiChunkFromAiSDK(deltaChunk1, context);
      const result3 = openaiChunkFromAiSDK(deltaChunk2, context);

      if (
        result1.type === "chunk" &&
        result2.type === "chunk" &&
        result3.type === "chunk"
      ) {
        expect(result1.chunk.choices[0].delta.tool_calls![0].index).toBe(0);
        expect(result2.chunk.choices[0].delta.tool_calls![0].index).toBe(0);
        expect(result3.chunk.choices[0].delta.tool_calls![0].index).toBe(0);
      }
    });

    it("converts complete tool-call chunk", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-call",
        toolCallId: "call_complete",
        toolName: "search",
        input: { query: "test" },
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("chunk");
      if (result.type === "chunk") {
        expect(result.chunk.choices[0].delta.tool_calls).toEqual([
          {
            index: 0,
            id: "call_complete",
            type: "function",
            function: {
              name: "search",
              arguments: '{"query":"test"}',
            },
          },
        ]);
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

      const result = openaiChunkFromAiSDK(chunk, context);

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

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });
  });

  describe("finish events", () => {
    it("converts finish-step with stop reason", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "finish-step",
        response: {
          id: "resp-1",
          timestamp: new Date(),
          modelId: "gpt-4o",
        },
        usage: {
          inputTokens: 10,
          outputTokens: 20,
          totalTokens: 30,
        },
        finishReason: "stop",
        providerMetadata: undefined,
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("chunk");
      if (result.type === "chunk") {
        expect(result.chunk.choices[0].finish_reason).toBe("stop");
        expect(result.chunk.choices[0].delta).toEqual({});
        expect(result.chunk.usage).toEqual({
          prompt_tokens: 10,
          completion_tokens: 20,
          total_tokens: 30,
        });
      }
    });

    it("converts finish-step with tool-calls reason", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "finish-step",
        response: {
          id: "resp-1",
          timestamp: new Date(),
          modelId: "gpt-4o",
        },
        usage: {
          inputTokens: 50,
          outputTokens: 25,
          totalTokens: 75,
        },
        finishReason: "tool-calls",
        providerMetadata: undefined,
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("chunk");
      if (result.type === "chunk") {
        expect(result.chunk.choices[0].finish_reason).toBe("tool_calls");
      }
    });

    it("converts finish-step with length reason", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "finish-step",
        response: {
          id: "resp-1",
          timestamp: new Date(),
          modelId: "gpt-4o",
        },
        usage: {
          inputTokens: 100,
          outputTokens: 4096,
          totalTokens: 4196,
        },
        finishReason: "length",
        providerMetadata: undefined,
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("chunk");
      if (result.type === "chunk") {
        expect(result.chunk.choices[0].finish_reason).toBe("length");
      }
    });

    it("converts finish-step with content-filter reason", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "finish-step",
        response: {
          id: "resp-1",
          timestamp: new Date(),
          modelId: "gpt-4o",
        },
        usage: {
          inputTokens: 10,
          outputTokens: 0,
          totalTokens: 10,
        },
        finishReason: "content-filter",
        providerMetadata: undefined,
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("chunk");
      if (result.type === "chunk") {
        expect(result.chunk.choices[0].finish_reason).toBe("content_filter");
      }
    });

    it("converts finish event", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "finish",
        finishReason: "stop",
        totalUsage: {
          inputTokens: 100,
          outputTokens: 50,
          totalTokens: 150,
        },
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("chunk");
      if (result.type === "chunk") {
        expect(result.chunk.choices[0].finish_reason).toBe("stop");
        expect(result.chunk.usage).toEqual({
          prompt_tokens: 100,
          completion_tokens: 50,
          total_tokens: 150,
        });
      }
    });

    it("handles undefined tokens in usage", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "finish",
        finishReason: "stop",
        totalUsage: {
          inputTokens: undefined,
          outputTokens: undefined,
          totalTokens: undefined,
        },
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("chunk");
      if (result.type === "chunk") {
        expect(result.chunk.usage).toEqual({
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
        });
      }
    });
  });

  describe("lifecycle events", () => {
    it("skips start", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "start",
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips start-step", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "start-step",
        request: {},
        warnings: [],
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips abort", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "abort",
      };

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });
  });

  describe("error handling", () => {
    it("converts error chunk to error result", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "error",
        error: new Error("Something went wrong"),
      };

      const result = openaiChunkFromAiSDK(chunk, context);

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

      const result = openaiChunkFromAiSDK(chunk, context);

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

      const result = openaiChunkFromAiSDK(chunk, context);

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

      const result = openaiChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips raw chunks", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "raw",
        rawValue: { some: "data" },
      };

      const result = openaiChunkFromAiSDK(chunk, context);

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
          response: { id: "r1", timestamp: new Date(), modelId: "gpt-4o" },
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

      const results = chunks.map((c) => openaiChunkFromAiSDK(c, context));

      // Filter to only chunks
      const openaiChunks = results
        .filter((r) => r.type === "chunk")
        .map((r) => (r as { type: "chunk"; chunk: unknown }).chunk);

      expect(openaiChunks.length).toBe(6); // text-start, 3x text-delta, finish-step, finish
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
          type: "finish-step",
          response: { id: "r1", timestamp: new Date(), modelId: "gpt-4o" },
          usage: { inputTokens: 20, outputTokens: 15, totalTokens: 35 },
          finishReason: "tool-calls",
          providerMetadata: undefined,
        },
      ];

      const results = chunks.map((c) => openaiChunkFromAiSDK(c, context));
      const openaiChunks = results.filter((r) => r.type === "chunk");

      // text-start, text-delta, tool-input-start, 2x tool-input-delta, finish-step
      expect(openaiChunks.length).toBe(6);

      // Check the tool call chunks have correct structure
      const toolStartResult = results[4];
      if (toolStartResult.type === "chunk") {
        expect(toolStartResult.chunk.choices[0].delta.tool_calls![0]).toMatchObject({
          index: 0,
          id: "call_1",
          type: "function",
          function: { name: "get_weather", arguments: "" },
        });
      }
    });
  });
});
