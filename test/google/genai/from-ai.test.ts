import { describe, it, expect } from "vitest";
import { googleGenaiFromAiSDK } from "~/google/genai/from-ai";
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
    id: "resp_123abc",
    timestamp: new Date("2024-01-15T10:30:00Z"),
    modelId: "gemini-2.0-flash",
    headers: {},
  },
  ...overrides,
});

describe("googleGenaiFromAiSDK", () => {
  describe("basic response fields", () => {
    it("converts response id", () => {
      const response = createMockResponse();

      const result = googleGenaiFromAiSDK(response);

      expect(result.responseId).toBe("resp_123abc");
    });

    it("allows overriding response id", () => {
      const response = createMockResponse();

      const result = googleGenaiFromAiSDK(response, { responseId: "custom_id" });

      expect(result.responseId).toBe("custom_id");
    });

    it("converts model version", () => {
      const response = createMockResponse({
        response: {
          id: "resp_123",
          timestamp: new Date(),
          modelId: "gemini-2.0-pro",
          headers: {},
        },
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.modelVersion).toBe("gemini-2.0-pro");
    });

    it("allows overriding model version", () => {
      const response = createMockResponse();

      const result = googleGenaiFromAiSDK(response, {
        modelVersion: "gemini-1.5-flash",
      });

      expect(result.modelVersion).toBe("gemini-1.5-flash");
    });

    it("creates candidates array with one candidate", () => {
      const response = createMockResponse();

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates).toHaveLength(1);
    });
  });

  describe("content", () => {
    it("converts text to text part", () => {
      const response = createMockResponse({
        text: "This is the response text",
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates![0].content?.parts).toHaveLength(1);
      expect(result.candidates![0].content?.parts![0]).toEqual({
        text: "This is the response text",
      });
    });

    it("sets role to model", () => {
      const response = createMockResponse();

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates![0].content?.role).toBe("model");
    });

    it("creates empty text part when no content", () => {
      const response = createMockResponse({
        text: "",
        toolCalls: [],
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates![0].content?.parts).toHaveLength(1);
      expect(result.candidates![0].content?.parts![0]).toEqual({ text: "" });
    });

    it("converts tool calls to function call parts", () => {
      const response = createMockResponse({
        text: "",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "call_abc123",
            toolName: "get_weather",
            input: { location: "San Francisco" },
          },
        ],
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates![0].content?.parts).toHaveLength(1);
      expect(result.candidates![0].content?.parts![0]).toEqual({
        functionCall: {
          id: "call_abc123",
          name: "get_weather",
          args: { location: "San Francisco" },
        },
      });
    });

    it("converts multiple tool calls", () => {
      const response = createMockResponse({
        text: "",
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

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates![0].content?.parts).toHaveLength(2);
      expect(result.candidates![0].content?.parts![0]).toMatchObject({
        functionCall: {
          id: "call_1",
          name: "tool_a",
        },
      });
      expect(result.candidates![0].content?.parts![1]).toMatchObject({
        functionCall: {
          id: "call_2",
          name: "tool_b",
        },
      });
    });

    it("includes both text and function call parts", () => {
      const response = createMockResponse({
        text: "Let me check that for you.",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "call_123",
            toolName: "search",
            input: { query: "test" },
          },
        ],
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates![0].content?.parts).toHaveLength(2);
      expect(result.candidates![0].content?.parts![0]).toEqual({
        text: "Let me check that for you.",
      });
      expect(result.candidates![0].content?.parts![1]).toMatchObject({
        functionCall: {
          id: "call_123",
          name: "search",
        },
      });
    });

    it("handles tool call with complex nested input", () => {
      const response = createMockResponse({
        text: "",
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

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates![0].content?.parts![0]).toEqual({
        functionCall: {
          id: "call_complex",
          name: "complex_tool",
          args: {
            nested: {
              array: [1, 2, 3],
              object: { key: "value" },
            },
            boolean: true,
            number: 42,
          },
        },
      });
    });
  });

  describe("finish reason", () => {
    it("converts stop to STOP", () => {
      const response = createMockResponse({
        finishReason: "stop",
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates![0].finishReason).toBe("STOP");
    });

    it("converts length to MAX_TOKENS", () => {
      const response = createMockResponse({
        finishReason: "length",
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates![0].finishReason).toBe("MAX_TOKENS");
    });

    it("converts content-filter to SAFETY", () => {
      const response = createMockResponse({
        finishReason: "content-filter",
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates![0].finishReason).toBe("SAFETY");
    });

    it("converts tool-calls to STOP", () => {
      const response = createMockResponse({
        finishReason: "tool-calls",
        toolCalls: [
          {
            type: "tool-call",
            toolCallId: "call_123",
            toolName: "test",
            input: {},
          },
        ],
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates![0].finishReason).toBe("STOP");
    });

    it("converts error to OTHER", () => {
      const response = createMockResponse({
        finishReason: "error",
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates![0].finishReason).toBe("OTHER");
    });

    it("converts other to OTHER", () => {
      const response = createMockResponse({
        finishReason: "other",
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates![0].finishReason).toBe("OTHER");
    });

    it("converts unknown to OTHER", () => {
      const response = createMockResponse({
        finishReason: "unknown",
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.candidates![0].finishReason).toBe("OTHER");
    });
  });

  describe("usage metadata", () => {
    it("converts usage tokens", () => {
      const response = createMockResponse({
        usage: {
          inputTokens: 100,
          outputTokens: 50,
          totalTokens: 150,
        },
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.usageMetadata?.promptTokenCount).toBe(100);
      expect(result.usageMetadata?.candidatesTokenCount).toBe(50);
      expect(result.usageMetadata?.totalTokenCount).toBe(150);
    });

    it("handles undefined inputTokens", () => {
      const response = createMockResponse({
        usage: {
          inputTokens: undefined,
          outputTokens: 50,
          totalTokens: 50,
        },
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.usageMetadata?.promptTokenCount).toBe(0);
    });

    it("handles undefined outputTokens", () => {
      const response = createMockResponse({
        usage: {
          inputTokens: 100,
          outputTokens: undefined,
          totalTokens: 100,
        },
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.usageMetadata?.candidatesTokenCount).toBe(0);
    });

    it("calculates totalTokenCount if not provided", () => {
      const response = createMockResponse({
        usage: {
          inputTokens: 100,
          outputTokens: 50,
          totalTokens: undefined,
        },
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result.usageMetadata?.totalTokenCount).toBe(150);
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
          id: "resp_weather123",
          timestamp: new Date("2024-06-15T14:30:00Z"),
          modelId: "gemini-2.0-flash",
          headers: {},
        },
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result).toMatchObject({
        responseId: "resp_weather123",
        modelVersion: "gemini-2.0-flash",
        candidates: [
          {
            content: {
              role: "model",
              parts: [
                { text: "The weather in San Francisco is sunny and 72°F." },
              ],
            },
            finishReason: "STOP",
          },
        ],
        usageMetadata: {
          promptTokenCount: 25,
          candidatesTokenCount: 15,
          totalTokenCount: 40,
        },
      });
    });

    it("converts a complete function call response", () => {
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
          id: "resp_tool456",
          timestamp: new Date("2024-06-15T14:35:00Z"),
          modelId: "gemini-2.0-flash",
          headers: {},
        },
      });

      const result = googleGenaiFromAiSDK(response);

      expect(result).toMatchObject({
        responseId: "resp_tool456",
        modelVersion: "gemini-2.0-flash",
        candidates: [
          {
            content: {
              role: "model",
              parts: [
                {
                  functionCall: {
                    id: "call_weather_123",
                    name: "get_current_weather",
                    args: {
                      location: "San Francisco, CA",
                      unit: "fahrenheit",
                    },
                  },
                },
              ],
            },
            finishReason: "STOP",
          },
        ],
        usageMetadata: {
          promptTokenCount: 50,
          candidatesTokenCount: 25,
          totalTokenCount: 75,
        },
      });
    });
  });
});
