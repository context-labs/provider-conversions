import { describe, it, expect } from "vitest";
import { anthropicMessagesFromAiSDK } from "~/anthropic/messages/from-ai";
import type { AiSDKResponse } from "~/ai";

const createMockResponse = (
  overrides: Partial<AiSDKResponse> = {},
): AiSDKResponse => ({
  text: "Hello, world!",
  reasoning: [],
  reasoningText: undefined,
  files: [],
  sources: [],
  toolCalls: [],
  toolResults: [],
  finishReason: "stop",
  usage: {
    inputTokens: 10,
    outputTokens: 20,
    totalTokens: 30,
  },
  response: {
    id: "msg_123abc",
    timestamp: new Date("2024-01-15T10:30:00Z"),
    modelId: "claude-sonnet-4-5-20250929",
    headers: {},
  },
  ...overrides,
});

describe("anthropicMessagesFromAiSDK", () => {
  describe("basic response fields", () => {
    it("converts response id", () => {
      const response = createMockResponse();

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.id).toBe("msg_123abc");
    });

    it("allows overriding response id", () => {
      const response = createMockResponse();

      const result = anthropicMessagesFromAiSDK(response, { id: "custom_id" });

      expect(result.id).toBe("custom_id");
    });

    it("sets type to message", () => {
      const response = createMockResponse();

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.type).toBe("message");
    });

    it("sets role to assistant", () => {
      const response = createMockResponse();

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.role).toBe("assistant");
    });

    it("converts model id", () => {
      const response = createMockResponse({
        response: {
          id: "msg_123",
          timestamp: new Date(),
          modelId: "claude-opus-4-5-20251101",
          headers: {},
        },
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.model).toBe("claude-opus-4-5-20251101");
    });

    it("allows overriding model", () => {
      const response = createMockResponse();

      const result = anthropicMessagesFromAiSDK(response, {
        model: "claude-3-5-haiku-latest",
      });

      expect(result.model).toBe("claude-3-5-haiku-latest");
    });

    it("sets stop_sequence to null", () => {
      const response = createMockResponse();

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.stop_sequence).toBeNull();
    });
  });

  describe("content", () => {
    it("converts text to text block", () => {
      const response = createMockResponse({
        text: "This is the response text",
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.content).toHaveLength(1);
      expect(result.content[0]).toEqual({
        type: "text",
        text: "This is the response text",
        citations: null,
      });
    });

    it("creates empty text block when no content", () => {
      const response = createMockResponse({
        text: "",
        toolCalls: [],
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.content).toHaveLength(1);
      expect(result.content[0]).toEqual({
        type: "text",
        text: "",
        citations: null,
      });
    });

    it("converts tool calls to tool_use blocks", () => {
      const response = createMockResponse({
        text: "",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "toolu_abc123",
            toolName: "get_weather",
            input: { location: "San Francisco" },
          },
        ],
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.content).toHaveLength(1);
      expect(result.content[0]).toEqual({
        type: "tool_use",
        id: "toolu_abc123",
        name: "get_weather",
        input: { location: "San Francisco" },
      });
    });

    it("converts multiple tool calls", () => {
      const response = createMockResponse({
        text: "",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "toolu_1",
            toolName: "tool_a",
            input: { param: "value1" },
          },
          {
            type: "tool-call",
            toolCallId: "toolu_2",
            toolName: "tool_b",
            input: { param: "value2" },
          },
        ],
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.content).toHaveLength(2);
      expect(result.content[0]).toMatchObject({
        type: "tool_use",
        id: "toolu_1",
        name: "tool_a",
      });
      expect(result.content[1]).toMatchObject({
        type: "tool_use",
        id: "toolu_2",
        name: "tool_b",
      });
    });

    it("includes both text and tool_use blocks", () => {
      const response = createMockResponse({
        text: "Let me check that for you.",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "toolu_123",
            toolName: "search",
            input: { query: "test" },
          },
        ],
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.content).toHaveLength(2);
      expect(result.content[0]).toEqual({
        type: "text",
        text: "Let me check that for you.",
        citations: null,
      });
      expect(result.content[1]).toMatchObject({
        type: "tool_use",
        id: "toolu_123",
      });
    });

    it("handles tool call with complex nested input", () => {
      const response = createMockResponse({
        text: "",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "toolu_complex",
            toolName: "complex_tool",
            input: {
              nested: {
                array: [1, 2, 3],
                object: { key: "value" },
              },
              boolean: true,
              number: 42,
            },
          },
        ],
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.content[0]).toEqual({
        type: "tool_use",
        id: "toolu_complex",
        name: "complex_tool",
        input: {
          nested: {
            array: [1, 2, 3],
            object: { key: "value" },
          },
          boolean: true,
          number: 42,
        },
      });
    });
  });

  describe("stop reason", () => {
    it("converts stop to end_turn", () => {
      const response = createMockResponse({
        finishReason: "stop",
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.stop_reason).toBe("end_turn");
    });

    it("converts length to max_tokens", () => {
      const response = createMockResponse({
        finishReason: "length",
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.stop_reason).toBe("max_tokens");
    });

    it("converts tool-calls to tool_use", () => {
      const response = createMockResponse({
        finishReason: "tool-calls",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "toolu_123",
            toolName: "test",
            input: {},
          },
        ],
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.stop_reason).toBe("tool_use");
    });

    it("converts content-filter to refusal", () => {
      const response = createMockResponse({
        finishReason: "content-filter",
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.stop_reason).toBe("refusal");
    });

    it("converts error to end_turn", () => {
      const response = createMockResponse({
        finishReason: "error",
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.stop_reason).toBe("end_turn");
    });

    it("converts other to end_turn", () => {
      const response = createMockResponse({
        finishReason: "other",
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.stop_reason).toBe("end_turn");
    });

    it("converts unknown to end_turn", () => {
      const response = createMockResponse({
        finishReason: "unknown",
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.stop_reason).toBe("end_turn");
    });
  });

  describe("usage", () => {
    it("converts usage tokens", () => {
      const response = createMockResponse({
        usage: {
          inputTokens: 100,
          outputTokens: 50,
          totalTokens: 150,
        },
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.usage.input_tokens).toBe(100);
      expect(result.usage.output_tokens).toBe(50);
    });

    it("handles undefined inputTokens", () => {
      const response = createMockResponse({
        usage: {
          inputTokens: undefined,
          outputTokens: 50,
          totalTokens: 50,
        },
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.usage.input_tokens).toBe(0);
    });

    it("handles undefined outputTokens", () => {
      const response = createMockResponse({
        usage: {
          inputTokens: 100,
          outputTokens: undefined,
          totalTokens: 100,
        },
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.usage.output_tokens).toBe(0);
    });

    it("sets cache tokens to null", () => {
      const response = createMockResponse();

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.usage.cache_creation_input_tokens).toBeNull();
      expect(result.usage.cache_read_input_tokens).toBeNull();
      expect(result.usage.cache_creation).toBeNull();
    });

    it("sets server_tool_use to null", () => {
      const response = createMockResponse();

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.usage.server_tool_use).toBeNull();
    });

    it("sets service_tier to null", () => {
      const response = createMockResponse();

      const result = anthropicMessagesFromAiSDK(response);

      expect(result.usage.service_tier).toBeNull();
    });
  });

  describe("full response conversion", () => {
    it("converts a complete text response", () => {
      const response = createMockResponse({
        text: "The weather in San Francisco is sunny and 72°F.",
        finishReason: "stop",
        usage: {
          inputTokens: 25,
          outputTokens: 15,
          totalTokens: 40,
        },
        response: {
          id: "msg_weather123",
          timestamp: new Date("2024-06-15T14:30:00Z"),
          modelId: "claude-sonnet-4-5-20250929",
          headers: {},
        },
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result).toMatchObject({
        id: "msg_weather123",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-5-20250929",
        content: [
          {
            type: "text",
            text: "The weather in San Francisco is sunny and 72°F.",
            citations: null,
          },
        ],
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: {
          input_tokens: 25,
          output_tokens: 15,
        },
      });
    });

    it("converts a complete tool call response", () => {
      const response = createMockResponse({
        text: "",
        finishReason: "tool-calls",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "toolu_weather_123",
            toolName: "get_current_weather",
            input: {
              location: "San Francisco, CA",
              unit: "fahrenheit",
            },
          },
        ],
        usage: {
          inputTokens: 50,
          outputTokens: 25,
          totalTokens: 75,
        },
        response: {
          id: "msg_tool456",
          timestamp: new Date("2024-06-15T14:35:00Z"),
          modelId: "claude-sonnet-4-5-20250929",
          headers: {},
        },
      });

      const result = anthropicMessagesFromAiSDK(response);

      expect(result).toMatchObject({
        id: "msg_tool456",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-5-20250929",
        content: [
          {
            type: "tool_use",
            id: "toolu_weather_123",
            name: "get_current_weather",
            input: {
              location: "San Francisco, CA",
              unit: "fahrenheit",
            },
          },
        ],
        stop_reason: "tool_use",
        stop_sequence: null,
        usage: {
          input_tokens: 50,
          output_tokens: 25,
        },
      });
    });
  });
});
