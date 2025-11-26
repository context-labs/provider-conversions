import { describe, it, expect } from "vitest";
import { openaiChatCompletionFromAiSDK } from "~/openai/chat-completion/from-ai";
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
    id: "resp_123",
    timestamp: new Date("2024-01-15T10:30:00Z"),
    modelId: "gpt-4o",
    headers: {},
  },
  ...overrides,
});

describe("openaiChatCompletionFromAiSDK", () => {
  describe("basic response fields", () => {
    it("converts response id", () => {
      const response = createMockResponse();

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.id).toBe("resp_123");
    });

    it("allows overriding response id", () => {
      const response = createMockResponse();

      const result = openaiChatCompletionFromAiSDK(response, {
        id: "custom_id",
      });

      expect(result.id).toBe("custom_id");
    });

    it("sets object to chat.completion", () => {
      const response = createMockResponse();

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.object).toBe("chat.completion");
    });

    it("converts timestamp to unix seconds", () => {
      const response = createMockResponse({
        response: {
          id: "resp_123",
          timestamp: new Date("2024-01-15T10:30:00Z"),
          modelId: "gpt-4o",
          headers: {},
        },
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.created).toBe(Math.floor(new Date("2024-01-15T10:30:00Z").getTime() / 1000));
    });

    it("allows overriding created timestamp", () => {
      const response = createMockResponse();

      const result = openaiChatCompletionFromAiSDK(response, {
        created: 1234567890,
      });

      expect(result.created).toBe(1234567890);
    });

    it("converts model id", () => {
      const response = createMockResponse({
        response: {
          id: "resp_123",
          timestamp: new Date(),
          modelId: "claude-3-opus",
          headers: {},
        },
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.model).toBe("claude-3-opus");
    });

    it("allows overriding model", () => {
      const response = createMockResponse();

      const result = openaiChatCompletionFromAiSDK(response, {
        model: "gpt-4-turbo",
      });

      expect(result.model).toBe("gpt-4-turbo");
    });
  });

  describe("message content", () => {
    it("converts text content", () => {
      const response = createMockResponse({
        text: "This is the response text",
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].message.content).toBe("This is the response text");
    });

    it("converts empty text to null", () => {
      const response = createMockResponse({
        text: "",
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].message.content).toBeNull();
    });

    it("sets role to assistant", () => {
      const response = createMockResponse();

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].message.role).toBe("assistant");
    });

    it("sets refusal to null", () => {
      const response = createMockResponse();

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].message.refusal).toBeNull();
    });

    it("sets choice index to 0", () => {
      const response = createMockResponse();

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].index).toBe(0);
    });

    it("sets logprobs to null", () => {
      const response = createMockResponse();

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].logprobs).toBeNull();
    });
  });

  describe("finish reason", () => {
    it("converts stop finish reason", () => {
      const response = createMockResponse({
        finishReason: "stop",
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].finish_reason).toBe("stop");
    });

    it("converts length finish reason", () => {
      const response = createMockResponse({
        finishReason: "length",
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].finish_reason).toBe("length");
    });

    it("converts content-filter to content_filter", () => {
      const response = createMockResponse({
        finishReason: "content-filter",
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].finish_reason).toBe("content_filter");
    });

    it("converts tool-calls to tool_calls", () => {
      const response = createMockResponse({
        finishReason: "tool-calls",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "call_123",
            toolName: "get_weather",
            input: { location: "SF" },
          },
        ],
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].finish_reason).toBe("tool_calls");
    });

    it("converts error finish reason to stop", () => {
      const response = createMockResponse({
        finishReason: "error",
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].finish_reason).toBe("stop");
    });

    it("converts other finish reason to stop", () => {
      const response = createMockResponse({
        finishReason: "other",
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].finish_reason).toBe("stop");
    });

    it("converts unknown finish reason to stop", () => {
      const response = createMockResponse({
        finishReason: "unknown",
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].finish_reason).toBe("stop");
    });
  });

  describe("tool calls", () => {
    it("converts single tool call", () => {
      const response = createMockResponse({
        text: "",
        finishReason: "tool-calls",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "call_abc123",
            toolName: "get_weather",
            input: { location: "San Francisco", unit: "celsius" },
          },
        ],
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].message.tool_calls).toEqual([
        {
          id: "call_abc123",
          type: "function",
          function: {
            name: "get_weather",
            arguments: '{"location":"San Francisco","unit":"celsius"}',
          },
        },
      ]);
    });

    it("converts multiple tool calls", () => {
      const response = createMockResponse({
        text: "",
        finishReason: "tool-calls",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "call_1",
            toolName: "tool_a",
            input: { param: "value1" },
          },
          {
            type: "tool-call",
            toolCallId: "call_2",
            toolName: "tool_b",
            input: { param: "value2" },
          },
        ],
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].message.tool_calls).toHaveLength(2);
      expect(result.choices[0].message.tool_calls![0].id).toBe("call_1");
      expect(result.choices[0].message.tool_calls![1].id).toBe("call_2");
    });

    it("does not include tool_calls when there are no tool calls", () => {
      const response = createMockResponse({
        toolCalls: [],
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].message).not.toHaveProperty("tool_calls");
    });

    it("handles tool call with complex nested input", () => {
      const response = createMockResponse({
        finishReason: "tool-calls",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "call_complex",
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

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].message.tool_calls![0].function.arguments).toBe(
        '{"nested":{"array":[1,2,3],"object":{"key":"value"}},"boolean":true,"number":42}',
      );
    });

    it("handles tool call with empty input", () => {
      const response = createMockResponse({
        finishReason: "tool-calls",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "call_empty",
            toolName: "no_params_tool",
            input: {},
          },
        ],
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].message.tool_calls![0].function.arguments).toBe("{}");
    });

    it("includes both content and tool_calls when both present", () => {
      const response = createMockResponse({
        text: "I'll help you with that.",
        finishReason: "tool-calls",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "call_123",
            toolName: "helper_tool",
            input: {},
          },
        ],
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.choices[0].message.content).toBe("I'll help you with that.");
      expect(result.choices[0].message.tool_calls).toHaveLength(1);
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

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.usage).toEqual({
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
      });
    });

    it("handles undefined inputTokens", () => {
      const response = createMockResponse({
        usage: {
          inputTokens: undefined,
          outputTokens: 50,
          totalTokens: 50,
        },
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.usage!.prompt_tokens).toBe(0);
    });

    it("handles undefined outputTokens", () => {
      const response = createMockResponse({
        usage: {
          inputTokens: 100,
          outputTokens: undefined,
          totalTokens: 100,
        },
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.usage!.completion_tokens).toBe(0);
    });

    it("calculates total_tokens when totalTokens is undefined", () => {
      const response = createMockResponse({
        usage: {
          inputTokens: 100,
          outputTokens: 50,
          totalTokens: undefined,
        },
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.usage!.total_tokens).toBe(150);
    });

    it("handles all undefined tokens", () => {
      const response = createMockResponse({
        usage: {
          inputTokens: undefined,
          outputTokens: undefined,
          totalTokens: undefined,
        },
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result.usage).toEqual({
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
      });
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
          id: "chatcmpl-abc123",
          timestamp: new Date("2024-06-15T14:30:00Z"),
          modelId: "gpt-4o-2024-05-13",
          headers: {},
        },
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result).toEqual({
        id: "chatcmpl-abc123",
        object: "chat.completion",
        created: Math.floor(new Date("2024-06-15T14:30:00Z").getTime() / 1000),
        model: "gpt-4o-2024-05-13",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "The weather in San Francisco is sunny and 72°F.",
              refusal: null,
            },
            logprobs: null,
            finish_reason: "stop",
          },
        ],
        usage: {
          prompt_tokens: 25,
          completion_tokens: 15,
          total_tokens: 40,
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
            toolCallId: "call_weather_123",
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
          id: "chatcmpl-tool456",
          timestamp: new Date("2024-06-15T14:35:00Z"),
          modelId: "gpt-4o",
          headers: {},
        },
      });

      const result = openaiChatCompletionFromAiSDK(response);

      expect(result).toEqual({
        id: "chatcmpl-tool456",
        object: "chat.completion",
        created: Math.floor(new Date("2024-06-15T14:35:00Z").getTime() / 1000),
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              refusal: null,
              tool_calls: [
                {
                  id: "call_weather_123",
                  type: "function",
                  function: {
                    name: "get_current_weather",
                    arguments: '{"location":"San Francisco, CA","unit":"fahrenheit"}',
                  },
                },
              ],
            },
            logprobs: null,
            finish_reason: "tool_calls",
          },
        ],
        usage: {
          prompt_tokens: 50,
          completion_tokens: 25,
          total_tokens: 75,
        },
      });
    });
  });
});
