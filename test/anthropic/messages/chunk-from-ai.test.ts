import { describe, it, expect, beforeEach } from "vitest";
import {
  anthropicChunkFromAiSDK,
  createChunkConversionContext,
  type ChunkConversionContext,
} from "~/anthropic/messages/chunk-from-ai";
import type { AiSDKChunk } from "~/ai";
import type { ToolSet } from "ai";

describe("anthropicChunkFromAiSDK", () => {
  let context: ChunkConversionContext;

  beforeEach(() => {
    context = createChunkConversionContext({
      id: "msg_123abc",
      model: "claude-sonnet-4-5-20250929",
    });
  });

  describe("createChunkConversionContext", () => {
    it("creates context with provided values", () => {
      const ctx = createChunkConversionContext({
        id: "test-id",
        model: "test-model",
      });

      expect(ctx.id).toBe("test-id");
      expect(ctx.model).toBe("test-model");
      expect(ctx.contentBlockIndex).toBe(0);
      expect(ctx.toolCallIndices.size).toBe(0);
      expect(ctx.toolCallInputs.size).toBe(0);
      expect(ctx.messageStarted).toBe(false);
      expect(ctx.inputTokens).toBe(0);
      expect(ctx.outputTokens).toBe(0);
    });
  });

  describe("text streaming", () => {
    it("converts text-start to message_start and content_block_start", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "text-start",
        id: "text-1",
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        expect(result.events).toHaveLength(2);

        // First event is message_start
        expect(result.events[0].type).toBe("message_start");
        if (result.events[0].type === "message_start") {
          expect(result.events[0].message.id).toBe("msg_123abc");
          expect(result.events[0].message.model).toBe("claude-sonnet-4-5-20250929");
          expect(result.events[0].message.role).toBe("assistant");
          expect(result.events[0].message.content).toEqual([]);
        }

        // Second event is content_block_start
        expect(result.events[1].type).toBe("content_block_start");
        if (result.events[1].type === "content_block_start") {
          expect(result.events[1].index).toBe(0);
          expect(result.events[1].content_block).toEqual({
            type: "text",
            text: "",
            citations: null,
          });
        }
      }
    });

    it("converts text-delta to content_block_delta", () => {
      // First start the text block
      const startChunk: AiSDKChunk<ToolSet> = {
        type: "text-start",
        id: "text-1",
      };
      anthropicChunkFromAiSDK(startChunk, context);

      // Then send delta
      const deltaChunk: AiSDKChunk<ToolSet> = {
        type: "text-delta",
        id: "text-1",
        text: "Hello, world!",
      };

      const result = anthropicChunkFromAiSDK(deltaChunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        expect(result.events).toHaveLength(1);
        expect(result.events[0].type).toBe("content_block_delta");
        if (result.events[0].type === "content_block_delta") {
          expect(result.events[0].index).toBe(0);
          expect(result.events[0].delta).toEqual({
            type: "text_delta",
            text: "Hello, world!",
          });
        }
      }
    });

    it("converts text-end to content_block_stop", () => {
      // Start the text block
      const startChunk: AiSDKChunk<ToolSet> = {
        type: "text-start",
        id: "text-1",
      };
      anthropicChunkFromAiSDK(startChunk, context);

      // End the text block
      const endChunk: AiSDKChunk<ToolSet> = {
        type: "text-end",
        id: "text-1",
      };

      const result = anthropicChunkFromAiSDK(endChunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        expect(result.events).toHaveLength(1);
        expect(result.events[0].type).toBe("content_block_stop");
        if (result.events[0].type === "content_block_stop") {
          expect(result.events[0].index).toBe(0);
        }
      }

      // Index should be incremented
      expect(context.contentBlockIndex).toBe(1);
    });
  });

  describe("reasoning streaming", () => {
    it("converts reasoning-start to message_start and content_block_start with thinking block", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "reasoning-start",
        id: "reasoning-1",
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        expect(result.events).toHaveLength(2);

        expect(result.events[0].type).toBe("message_start");
        expect(result.events[1].type).toBe("content_block_start");
        if (result.events[1].type === "content_block_start") {
          expect(result.events[1].content_block).toMatchObject({
            type: "thinking",
            thinking: "",
            signature: "",
          });
        }
      }
    });

    it("converts reasoning-delta to thinking_delta", () => {
      // Start reasoning block
      const startChunk: AiSDKChunk<ToolSet> = {
        type: "reasoning-start",
        id: "reasoning-1",
      };
      anthropicChunkFromAiSDK(startChunk, context);

      // Send delta
      const deltaChunk: AiSDKChunk<ToolSet> = {
        type: "reasoning-delta",
        id: "reasoning-1",
        text: "Let me think about this...",
      };

      const result = anthropicChunkFromAiSDK(deltaChunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        expect(result.events).toHaveLength(1);
        expect(result.events[0].type).toBe("content_block_delta");
        if (result.events[0].type === "content_block_delta") {
          expect(result.events[0].delta).toEqual({
            type: "thinking_delta",
            thinking: "Let me think about this...",
          });
        }
      }
    });

    it("converts reasoning-end to content_block_stop", () => {
      // Start reasoning block
      const startChunk: AiSDKChunk<ToolSet> = {
        type: "reasoning-start",
        id: "reasoning-1",
      };
      anthropicChunkFromAiSDK(startChunk, context);

      // End reasoning block
      const endChunk: AiSDKChunk<ToolSet> = {
        type: "reasoning-end",
        id: "reasoning-1",
      };

      const result = anthropicChunkFromAiSDK(endChunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        expect(result.events).toHaveLength(1);
        expect(result.events[0].type).toBe("content_block_stop");
      }
    });
  });

  describe("tool call streaming", () => {
    it("converts tool-input-start to message_start and content_block_start with tool_use block", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-start",
        id: "toolu_abc123",
        toolName: "get_weather",
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        expect(result.events).toHaveLength(2);

        expect(result.events[0].type).toBe("message_start");
        expect(result.events[1].type).toBe("content_block_start");
        if (result.events[1].type === "content_block_start") {
          expect(result.events[1].content_block).toEqual({
            type: "tool_use",
            id: "toolu_abc123",
            name: "get_weather",
            input: {},
          });
        }
      }
    });

    it("converts tool-input-delta to input_json_delta", () => {
      // Start tool call
      const startChunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-start",
        id: "toolu_abc123",
        toolName: "get_weather",
      };
      anthropicChunkFromAiSDK(startChunk, context);

      // Send delta
      const deltaChunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-delta",
        id: "toolu_abc123",
        delta: '{"location":',
      };

      const result = anthropicChunkFromAiSDK(deltaChunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        expect(result.events).toHaveLength(1);
        expect(result.events[0].type).toBe("content_block_delta");
        if (result.events[0].type === "content_block_delta") {
          expect(result.events[0].index).toBe(0);
          expect(result.events[0].delta).toEqual({
            type: "input_json_delta",
            partial_json: '{"location":',
          });
        }
      }
    });

    it("returns error for tool-input-delta with unknown id", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-delta",
        id: "unknown_tool",
        delta: "{}",
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("error");
      if (result.type === "error") {
        expect(result.error).toContain("Unknown tool call id");
      }
    });

    it("converts tool-input-end to content_block_stop", () => {
      // Start tool call
      const startChunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-start",
        id: "toolu_abc123",
        toolName: "get_weather",
      };
      anthropicChunkFromAiSDK(startChunk, context);

      // End tool call
      const endChunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-end",
        id: "toolu_abc123",
      };

      const result = anthropicChunkFromAiSDK(endChunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        expect(result.events).toHaveLength(1);
        expect(result.events[0].type).toBe("content_block_stop");
      }
    });

    it("returns error for tool-input-end with unknown id", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-input-end",
        id: "unknown_tool",
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("error");
      if (result.type === "error") {
        expect(result.error).toContain("Unknown tool call id");
      }
    });

    it("converts complete tool-call to start, delta, and stop events", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-call",
        toolCallId: "toolu_complete",
        toolName: "search",
        input: { query: "test" },
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        expect(result.events).toHaveLength(4);

        // message_start
        expect(result.events[0].type).toBe("message_start");

        // content_block_start
        expect(result.events[1].type).toBe("content_block_start");
        if (result.events[1].type === "content_block_start") {
          expect(result.events[1].content_block).toEqual({
            type: "tool_use",
            id: "toolu_complete",
            name: "search",
            input: {},
          });
        }

        // content_block_delta with full input
        expect(result.events[2].type).toBe("content_block_delta");
        if (result.events[2].type === "content_block_delta") {
          expect(result.events[2].delta).toEqual({
            type: "input_json_delta",
            partial_json: '{"query":"test"}',
          });
        }

        // content_block_stop
        expect(result.events[3].type).toBe("content_block_stop");
      }
    });

    it("skips tool-result", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-result",
        toolCallId: "toolu_123",
        toolName: "get_weather",
        output: { type: "text", value: "Sunny" },
        input: {},
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips tool-error", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "tool-error",
        toolCallId: "toolu_123",
        toolName: "failing_tool",
        input: {},
        error: new Error("Tool failed"),
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

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
          modelId: "claude-sonnet-4-5-20250929",
        },
        usage: {
          inputTokens: 100,
          outputTokens: 50,
          totalTokens: 150,
        },
        finishReason: "stop",
        providerMetadata: undefined,
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
      expect(context.inputTokens).toBe(100);
      expect(context.outputTokens).toBe(50);
    });

    it("converts finish to message_delta and message_stop", () => {
      // First emit message_start
      const startChunk: AiSDKChunk<ToolSet> = {
        type: "text-start",
        id: "text-1",
      };
      anthropicChunkFromAiSDK(startChunk, context);

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

      const result = anthropicChunkFromAiSDK(finishChunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        expect(result.events).toHaveLength(2);

        // message_delta
        expect(result.events[0].type).toBe("message_delta");
        if (result.events[0].type === "message_delta") {
          expect(result.events[0].delta.stop_reason).toBe("end_turn");
          expect(result.events[0].delta.stop_sequence).toBeNull();
          expect(result.events[0].usage.output_tokens).toBe(50);
        }

        // message_stop
        expect(result.events[1].type).toBe("message_stop");
      }
    });

    it("converts tool-calls finish reason", () => {
      // Start message
      const startChunk: AiSDKChunk<ToolSet> = {
        type: "start",
      };
      anthropicChunkFromAiSDK(startChunk, context);

      const chunk: AiSDKChunk<ToolSet> = {
        type: "finish",
        finishReason: "tool-calls",
        totalUsage: {
          inputTokens: 50,
          outputTokens: 25,
          totalTokens: 75,
        },
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        const deltaEvent = result.events.find((e) => e.type === "message_delta");
        expect(deltaEvent).toBeDefined();
        if (deltaEvent?.type === "message_delta") {
          expect(deltaEvent.delta.stop_reason).toBe("tool_use");
        }
      }
    });

    it("converts length finish reason to max_tokens", () => {
      const startChunk: AiSDKChunk<ToolSet> = {
        type: "start",
      };
      anthropicChunkFromAiSDK(startChunk, context);

      const chunk: AiSDKChunk<ToolSet> = {
        type: "finish",
        finishReason: "length",
        totalUsage: {
          inputTokens: 100,
          outputTokens: 4096,
          totalTokens: 4196,
        },
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        const deltaEvent = result.events.find((e) => e.type === "message_delta");
        if (deltaEvent?.type === "message_delta") {
          expect(deltaEvent.delta.stop_reason).toBe("max_tokens");
        }
      }
    });

    it("converts content-filter finish reason to refusal", () => {
      const startChunk: AiSDKChunk<ToolSet> = {
        type: "start",
      };
      anthropicChunkFromAiSDK(startChunk, context);

      const chunk: AiSDKChunk<ToolSet> = {
        type: "finish",
        finishReason: "content-filter",
        totalUsage: {
          inputTokens: 10,
          outputTokens: 0,
          totalTokens: 10,
        },
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        const deltaEvent = result.events.find((e) => e.type === "message_delta");
        if (deltaEvent?.type === "message_delta") {
          expect(deltaEvent.delta.stop_reason).toBe("refusal");
        }
      }
    });

    it("emits message_start if not already started on finish", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "finish",
        finishReason: "stop",
        totalUsage: {
          inputTokens: 0,
          outputTokens: 0,
          totalTokens: 0,
        },
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        expect(result.events).toHaveLength(3);
        expect(result.events[0].type).toBe("message_start");
        expect(result.events[1].type).toBe("message_delta");
        expect(result.events[2].type).toBe("message_stop");
      }
    });
  });

  describe("lifecycle events", () => {
    it("converts start to message_start", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "start",
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("events");
      if (result.type === "events") {
        expect(result.events).toHaveLength(1);
        expect(result.events[0].type).toBe("message_start");
      }
    });

    it("only emits message_start once", () => {
      const chunk1: AiSDKChunk<ToolSet> = {
        type: "start",
      };
      const chunk2: AiSDKChunk<ToolSet> = {
        type: "text-start",
        id: "text-1",
      };

      const result1 = anthropicChunkFromAiSDK(chunk1, context);
      const result2 = anthropicChunkFromAiSDK(chunk2, context);

      expect(result1.type).toBe("events");
      if (result1.type === "events") {
        expect(result1.events).toHaveLength(1);
        expect(result1.events[0].type).toBe("message_start");
      }

      expect(result2.type).toBe("events");
      if (result2.type === "events") {
        // Should only have content_block_start, no message_start
        expect(result2.events).toHaveLength(1);
        expect(result2.events[0].type).toBe("content_block_start");
      }
    });

    it("skips start-step", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "start-step",
        request: {},
        warnings: [],
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips abort", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "abort",
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });
  });

  describe("error handling", () => {
    it("converts error chunk to error result", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "error",
        error: new Error("Something went wrong"),
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

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

      const result = anthropicChunkFromAiSDK(chunk, context);

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

      const result = anthropicChunkFromAiSDK(chunk, context);

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

      const result = anthropicChunkFromAiSDK(chunk, context);

      expect(result.type).toBe("skip");
    });

    it("skips raw chunks", () => {
      const chunk: AiSDKChunk<ToolSet> = {
        type: "raw",
        rawValue: { some: "data" },
      };

      const result = anthropicChunkFromAiSDK(chunk, context);

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
          response: { id: "r1", timestamp: new Date(), modelId: "claude-sonnet-4-5-20250929" },
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

      const allEvents: Array<{ type: string }> = [];
      for (const chunk of chunks) {
        const result = anthropicChunkFromAiSDK(chunk, context);
        if (result.type === "events") {
          allEvents.push(...result.events);
        }
      }

      // Expected sequence:
      // 1. message_start (from start)
      // 2. content_block_start (from text-start, no message_start since already done)
      // 3. content_block_delta x3 (from text-delta)
      // 4. content_block_stop (from text-end)
      // 5. message_delta (from finish)
      // 6. message_stop (from finish)

      expect(allEvents.map((e) => e.type)).toEqual([
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
      ]);
    });

    it("handles a complete tool call streaming sequence", () => {
      const chunks: AiSDKChunk<ToolSet>[] = [
        { type: "start" },
        { type: "text-start", id: "text-1" },
        { type: "text-delta", id: "text-1", text: "Let me check the weather." },
        { type: "text-end", id: "text-1" },
        { type: "tool-input-start", id: "toolu_123", toolName: "get_weather" },
        { type: "tool-input-delta", id: "toolu_123", delta: '{"location"' },
        { type: "tool-input-delta", id: "toolu_123", delta: ':"San Francisco"}' },
        { type: "tool-input-end", id: "toolu_123" },
        {
          type: "finish",
          finishReason: "tool-calls",
          totalUsage: { inputTokens: 20, outputTokens: 15, totalTokens: 35 },
        },
      ];

      const allEvents: Array<{ type: string }> = [];
      for (const chunk of chunks) {
        const result = anthropicChunkFromAiSDK(chunk, context);
        if (result.type === "events") {
          allEvents.push(...result.events);
        }
      }

      expect(allEvents.map((e) => e.type)).toEqual([
        "message_start",
        "content_block_start",  // text
        "content_block_delta",   // text delta
        "content_block_stop",    // text end
        "content_block_start",   // tool_use
        "content_block_delta",   // input_json_delta
        "content_block_delta",   // input_json_delta
        "content_block_stop",    // tool_use end
        "message_delta",
        "message_stop",
      ]);

      // Verify tool_use content block
      const toolStartEvent = allEvents[4];
      if ("content_block" in toolStartEvent) {
        expect(toolStartEvent.content_block).toMatchObject({
          type: "tool_use",
          id: "toolu_123",
          name: "get_weather",
        });
      }
    });

    it("handles thinking followed by text", () => {
      const chunks: AiSDKChunk<ToolSet>[] = [
        { type: "start" },
        { type: "reasoning-start", id: "thinking-1" },
        { type: "reasoning-delta", id: "thinking-1", text: "I need to analyze this..." },
        { type: "reasoning-end", id: "thinking-1" },
        { type: "text-start", id: "text-1" },
        { type: "text-delta", id: "text-1", text: "Here's my answer." },
        { type: "text-end", id: "text-1" },
        {
          type: "finish",
          finishReason: "stop",
          totalUsage: { inputTokens: 50, outputTokens: 30, totalTokens: 80 },
        },
      ];

      const allEvents: Array<{ type: string }> = [];
      for (const chunk of chunks) {
        const result = anthropicChunkFromAiSDK(chunk, context);
        if (result.type === "events") {
          allEvents.push(...result.events);
        }
      }

      expect(allEvents.map((e) => e.type)).toEqual([
        "message_start",
        "content_block_start",  // thinking
        "content_block_delta",   // thinking delta
        "content_block_stop",    // thinking end
        "content_block_start",   // text
        "content_block_delta",   // text delta
        "content_block_stop",    // text end
        "message_delta",
        "message_stop",
      ]);

      // Verify thinking content block at index 0
      const thinkingStartEvent = allEvents[1];
      if ("content_block" in thinkingStartEvent) {
        expect(thinkingStartEvent.content_block).toMatchObject({
          type: "thinking",
        });
      }

      // Verify text content block at index 1
      const textStartEvent = allEvents[4];
      if ("content_block" in textStartEvent && "index" in textStartEvent) {
        expect(textStartEvent.index).toBe(1);
        expect(textStartEvent.content_block).toMatchObject({
          type: "text",
        });
      }
    });
  });
});
