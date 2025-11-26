import { describe, it, expect } from "vitest";
import { openaiChatCompletionToAiSDK } from "~/openai/chat-completion/to-ai";
import type { ChatCompletionRequestBody } from "~/openai/chat-completion/types";

describe("openaiChatCompletionToAiSDK", () => {
  describe("basic params", () => {
    it("converts model", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.model).toBe("gpt-4o");
    });

    it("converts max_completion_tokens", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        max_completion_tokens: 1000,
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.maxOutputTokens).toBe(1000);
    });

    it("converts max_tokens (deprecated) when max_completion_tokens is not set", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        max_tokens: 500,
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.maxOutputTokens).toBe(500);
    });

    it("prefers max_completion_tokens over max_tokens", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        max_completion_tokens: 1000,
        max_tokens: 500,
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.maxOutputTokens).toBe(1000);
    });

    it("converts temperature", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        temperature: 0.7,
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.temperature).toBe(0.7);
    });

    it("converts top_p to topP", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        top_p: 0.9,
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.topP).toBe(0.9);
    });

    it("converts frequency_penalty to frequencyPenalty", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        frequency_penalty: 0.5,
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.frequencyPenalty).toBe(0.5);
    });

    it("converts presence_penalty to presencePenalty", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        presence_penalty: 0.3,
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.presencePenalty).toBe(0.3);
    });

    it("converts seed", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        seed: 12345,
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.seed).toBe(12345);
    });
  });

  describe("stop sequences", () => {
    it("converts single stop string to array", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        stop: "END",
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.stopSequences).toEqual(["END"]);
    });

    it("passes through stop array", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        stop: ["END", "STOP"],
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.stopSequences).toEqual(["END", "STOP"]);
    });

    it("returns undefined for null stop", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        stop: null,
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.stopSequences).toBeUndefined();
    });
  });

  describe("messages", () => {
    describe("system messages", () => {
      it("converts system message with string content", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            { role: "system", content: "You are a helpful assistant" },
            { role: "user", content: "Hello" },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "system",
          content: "You are a helpful assistant",
        });
      });

      it("converts developer message to system message", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            { role: "developer", content: "You are a helpful assistant" },
            { role: "user", content: "Hello" },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "system",
          content: "You are a helpful assistant",
        });
      });
    });

    describe("user messages", () => {
      it("converts user message with string content", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [{ role: "user", content: "Hello, world!" }],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "user",
          content: "Hello, world!",
        });
      });

      it("converts user message with text content parts", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "First part" },
                { type: "text", text: "Second part" },
              ],
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "user",
          content: [
            { type: "text", text: "First part" },
            { type: "text", text: "Second part" },
          ],
        });
      });

      it("converts user message with image_url content", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "What is in this image?" },
                {
                  type: "image_url",
                  image_url: { url: "https://example.com/image.png" },
                },
              ],
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "user",
          content: [
            { type: "text", text: "What is in this image?" },
            { type: "image", image: "https://example.com/image.png" },
          ],
        });
      });

      it("converts user message with base64 image", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "image_url",
                  image_url: { url: "data:image/png;base64,iVBORw0KGgo=" },
                },
              ],
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "user",
          content: [
            { type: "image", image: "data:image/png;base64,iVBORw0KGgo=" },
          ],
        });
      });

      it("converts input_audio to empty text fallback", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "input_audio",
                  input_audio: { data: "base64audio", format: "wav" },
                },
              ],
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "user",
          content: [{ type: "text", text: "" }],
        });
      });

      it("converts file content part with file_data", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "file",
                  file: {
                    file_data: "base64filedata",
                    filename: "test.txt",
                  },
                },
              ],
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "user",
          content: [
            {
              type: "file",
              data: "base64filedata",
              mediaType: "application/octet-stream",
            },
          ],
        });
      });

      it("converts file content part with only file_id to empty text", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "file",
                  file: {
                    file_id: "file-123",
                  },
                },
              ],
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "user",
          content: [{ type: "text", text: "" }],
        });
      });
    });

    describe("assistant messages", () => {
      it("converts assistant message with string content", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            { role: "user", content: "Hello" },
            { role: "assistant", content: "Hi there!" },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[1]).toEqual({
          role: "assistant",
          content: [{ type: "text", text: "Hi there!" }],
        });
      });

      it("converts assistant message with content parts", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            { role: "user", content: "Hello" },
            {
              role: "assistant",
              content: [
                { type: "text", text: "First response" },
                { type: "text", text: "Second response" },
              ],
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[1]).toEqual({
          role: "assistant",
          content: [
            { type: "text", text: "First response" },
            { type: "text", text: "Second response" },
          ],
        });
      });

      it("converts assistant message with tool calls", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            { role: "user", content: "What is the weather?" },
            {
              role: "assistant",
              tool_calls: [
                {
                  id: "call_123",
                  type: "function",
                  function: {
                    name: "get_weather",
                    arguments: '{"location": "San Francisco"}',
                  },
                },
              ],
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[1]).toEqual({
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "call_123",
              toolName: "get_weather",
              input: { location: "San Francisco" },
            },
          ],
        });
      });

      it("converts assistant message with both content and tool calls", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            { role: "user", content: "What is the weather?" },
            {
              role: "assistant",
              content: "Let me check the weather for you.",
              tool_calls: [
                {
                  id: "call_123",
                  type: "function",
                  function: {
                    name: "get_weather",
                    arguments: '{"location": "San Francisco"}',
                  },
                },
              ],
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[1]).toEqual({
          role: "assistant",
          content: [
            { type: "text", text: "Let me check the weather for you." },
            {
              type: "tool-call",
              toolCallId: "call_123",
              toolName: "get_weather",
              input: { location: "San Francisco" },
            },
          ],
        });
      });

      it("handles invalid JSON in tool call arguments gracefully", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            { role: "user", content: "Hello" },
            {
              role: "assistant",
              tool_calls: [
                {
                  id: "call_123",
                  type: "function",
                  function: {
                    name: "my_tool",
                    arguments: "invalid json {",
                  },
                },
              ],
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[1]).toEqual({
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "call_123",
              toolName: "my_tool",
              input: "invalid json {",
            },
          ],
        });
      });

      it("skips refusal content parts", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            { role: "user", content: "Hello" },
            {
              role: "assistant",
              content: [
                { type: "text", text: "Sure!" },
                { type: "refusal", refusal: "I cannot do that" },
              ],
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[1]).toEqual({
          role: "assistant",
          content: [{ type: "text", text: "Sure!" }],
        });
      });
    });

    describe("tool messages", () => {
      it("converts tool message with string content", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            { role: "user", content: "What is the weather?" },
            {
              role: "assistant",
              tool_calls: [
                {
                  id: "call_123",
                  type: "function",
                  function: {
                    name: "get_weather",
                    arguments: '{"location": "SF"}',
                  },
                },
              ],
            },
            {
              role: "tool",
              tool_call_id: "call_123",
              content: "Sunny, 72°F",
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[2]).toEqual({
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_123",
              toolName: "",
              output: { type: "text", value: "Sunny, 72°F" },
            },
          ],
        });
      });

      it("converts tool message with array content", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            { role: "user", content: "Hello" },
            {
              role: "tool",
              tool_call_id: "call_456",
              content: [
                { type: "text", text: "Part 1" },
                { type: "text", text: "Part 2" },
              ],
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[1]).toEqual({
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_456",
              toolName: "",
              output: { type: "text", value: "Part 1Part 2" },
            },
          ],
        });
      });
    });

    describe("function messages (deprecated)", () => {
      it("converts function message to tool message", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            { role: "user", content: "Hello" },
            {
              role: "function",
              name: "my_function",
              content: "Function result",
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[1]).toEqual({
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "my_function",
              toolName: "my_function",
              output: { type: "text", value: "Function result" },
            },
          ],
        });
      });

      it("handles function message with null content", () => {
        const input: ChatCompletionRequestBody = {
          model: "gpt-4o",
          messages: [
            { role: "user", content: "Hello" },
            {
              role: "function",
              name: "my_function",
              content: null,
            },
          ],
        };

        const result = openaiChatCompletionToAiSDK(input);

        expect(result.messages[1]).toEqual({
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "my_function",
              toolName: "my_function",
              output: { type: "text", value: "" },
            },
          ],
        });
      });
    });
  });

  describe("tools", () => {
    it("converts function tools", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        tools: [
          {
            type: "function",
            function: {
              name: "get_weather",
              description: "Get weather for a location",
              parameters: {
                type: "object",
                properties: {
                  location: { type: "string" },
                },
                required: ["location"],
              },
            },
          },
        ],
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.tools).toEqual({
        get_weather: {
          description: "Get weather for a location",
          inputSchema: {
            type: "object",
            properties: {
              location: { type: "string" },
            },
            required: ["location"],
          },
        },
      });
    });

    it("converts multiple tools", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        tools: [
          {
            type: "function",
            function: {
              name: "tool_a",
              description: "Tool A",
            },
          },
          {
            type: "function",
            function: {
              name: "tool_b",
              description: "Tool B",
              parameters: { type: "object" },
            },
          },
        ],
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.tools).toEqual({
        tool_a: {
          description: "Tool A",
          inputSchema: {},
        },
        tool_b: {
          description: "Tool B",
          inputSchema: { type: "object" },
        },
      });
    });

    it("returns undefined for empty tools array", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        tools: [],
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.tools).toBeUndefined();
    });

    it("returns undefined when tools is not provided", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.tools).toBeUndefined();
    });
  });

  describe("tool choice", () => {
    it("converts 'none' tool choice", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        tool_choice: "none",
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.toolChoice).toBe("none");
    });

    it("converts 'auto' tool choice", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        tool_choice: "auto",
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.toolChoice).toBe("auto");
    });

    it("converts 'required' tool choice", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        tool_choice: "required",
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.toolChoice).toBe("required");
    });

    it("converts named function tool choice", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        tool_choice: {
          type: "function",
          function: { name: "get_weather" },
        },
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.toolChoice).toEqual({
        type: "tool",
        toolName: "get_weather",
      });
    });

    it("converts named custom tool choice", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        tool_choice: {
          type: "custom",
          custom: { name: "my_custom_tool" },
        },
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.toolChoice).toEqual({
        type: "tool",
        toolName: "my_custom_tool",
      });
    });

    it("converts allowed_tools choice to auto fallback", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        tool_choice: {
          type: "allowed_tools",
          allowed_tools: [{ type: "function", name: "tool1" }],
        },
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.toolChoice).toBe("auto");
    });

    it("returns undefined when tool_choice is not provided", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result.toolChoice).toBeUndefined();
    });
  });

  describe("full conversation", () => {
    it("converts a complete multi-turn conversation with tools", () => {
      const input: ChatCompletionRequestBody = {
        model: "gpt-4o",
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: "What's the weather in San Francisco?" },
          {
            role: "assistant",
            content: "I'll check the weather for you.",
            tool_calls: [
              {
                id: "call_abc123",
                type: "function",
                function: {
                  name: "get_weather",
                  arguments: '{"location": "San Francisco", "unit": "fahrenheit"}',
                },
              },
            ],
          },
          {
            role: "tool",
            tool_call_id: "call_abc123",
            content: '{"temperature": 72, "condition": "sunny"}',
          },
          {
            role: "assistant",
            content: "The weather in San Francisco is sunny with a temperature of 72°F.",
          },
        ],
        tools: [
          {
            type: "function",
            function: {
              name: "get_weather",
              description: "Get the current weather",
              parameters: {
                type: "object",
                properties: {
                  location: { type: "string" },
                  unit: { type: "string", enum: ["celsius", "fahrenheit"] },
                },
                required: ["location"],
              },
            },
          },
        ],
        temperature: 0.7,
        max_completion_tokens: 1000,
      };

      const result = openaiChatCompletionToAiSDK(input);

      expect(result).toEqual({
        model: "gpt-4o",
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: "What's the weather in San Francisco?" },
          {
            role: "assistant",
            content: [
              { type: "text", text: "I'll check the weather for you." },
              {
                type: "tool-call",
                toolCallId: "call_abc123",
                toolName: "get_weather",
                input: { location: "San Francisco", unit: "fahrenheit" },
              },
            ],
          },
          {
            role: "tool",
            content: [
              {
                type: "tool-result",
                toolCallId: "call_abc123",
                toolName: "",
                output: {
                  type: "text",
                  value: '{"temperature": 72, "condition": "sunny"}',
                },
              },
            ],
          },
          {
            role: "assistant",
            content: [
              {
                type: "text",
                text: "The weather in San Francisco is sunny with a temperature of 72°F.",
              },
            ],
          },
        ],
        tools: {
          get_weather: {
            description: "Get the current weather",
            inputSchema: {
              type: "object",
              properties: {
                location: { type: "string" },
                unit: { type: "string", enum: ["celsius", "fahrenheit"] },
              },
              required: ["location"],
            },
          },
        },
        maxOutputTokens: 1000,
        temperature: 0.7,
        topP: undefined,
        frequencyPenalty: undefined,
        presencePenalty: undefined,
        stopSequences: undefined,
        seed: undefined,
        toolChoice: undefined,
      });
    });
  });
});
