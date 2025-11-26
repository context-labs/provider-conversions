import { describe, it, expect } from "vitest";
import { anthropicMessagesToAiSDK } from "~/anthropic/messages/to-ai";
import type { MessageCreateParams } from "~/anthropic/messages/types";

describe("anthropicMessagesToAiSDK", () => {
  describe("basic params", () => {
    it("converts model", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.model).toBe("claude-sonnet-4-5-20250929");
    });

    it("converts max_tokens to maxOutputTokens", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 2048,
        messages: [{ role: "user", content: "Hello" }],
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.maxOutputTokens).toBe(2048);
    });

    it("converts temperature", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
        temperature: 0.7,
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.temperature).toBe(0.7);
    });

    it("converts top_p to topP", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
        top_p: 0.9,
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.topP).toBe(0.9);
    });

    it("converts top_k to topK", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
        top_k: 40,
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.topK).toBe(40);
    });

    it("converts stop_sequences to stopSequences", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
        stop_sequences: ["END", "STOP"],
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.stopSequences).toEqual(["END", "STOP"]);
    });
  });

  describe("system prompt", () => {
    it("converts string system prompt", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
        system: "You are a helpful assistant.",
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.system).toBe("You are a helpful assistant.");
    });

    it("converts array system prompt by joining text", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
        system: [
          { type: "text", text: "You are a helpful assistant." },
          { type: "text", text: "Be concise." },
        ],
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.system).toBe("You are a helpful assistant.\nBe concise.");
    });

    it("returns undefined for missing system prompt", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.system).toBeUndefined();
    });
  });

  describe("messages", () => {
    describe("user messages", () => {
      it("converts user message with string content", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
          messages: [{ role: "user", content: "Hello, world!" }],
        };

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "user",
          content: "Hello, world!",
        });
      });

      it("converts user message with text content blocks", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
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

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "user",
          content: [
            { type: "text", text: "First part" },
            { type: "text", text: "Second part" },
          ],
        });
      });

      it("converts user message with base64 image", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "What is this?" },
                {
                  type: "image",
                  source: {
                    type: "base64",
                    media_type: "image/png",
                    data: "iVBORw0KGgo=",
                  },
                },
              ],
            },
          ],
        };

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "user",
          content: [
            { type: "text", text: "What is this?" },
            { type: "image", image: "data:image/png;base64,iVBORw0KGgo=" },
          ],
        });
      });

      it("converts user message with URL image", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "image",
                  source: {
                    type: "url",
                    url: "https://example.com/image.png",
                  },
                },
              ],
            },
          ],
        };

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "user",
          content: [{ type: "image", image: "https://example.com/image.png" }],
        });
      });

      it("converts user message with base64 PDF document", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "document",
                  source: {
                    type: "base64",
                    media_type: "application/pdf",
                    data: "JVBERi0xLjQ=",
                  },
                },
              ],
            },
          ],
        };

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "user",
          content: [
            { type: "file", data: "JVBERi0xLjQ=", mediaType: "application/pdf" },
          ],
        });
      });

      it("converts user message with plain text document", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "document",
                  source: {
                    type: "text",
                    media_type: "text/plain",
                    data: "This is plain text content",
                  },
                },
              ],
            },
          ],
        };

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages[0]).toEqual({
          role: "user",
          content: [{ type: "text", text: "This is plain text content" }],
        });
      });
    });

    describe("assistant messages", () => {
      it("converts assistant message with string content", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
          messages: [
            { role: "user", content: "Hello" },
            { role: "assistant", content: "Hi there!" },
          ],
        };

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages[1]).toEqual({
          role: "assistant",
          content: [{ type: "text", text: "Hi there!" }],
        });
      });

      it("converts assistant message with text content blocks", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
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

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages[1]).toEqual({
          role: "assistant",
          content: [
            { type: "text", text: "First response" },
            { type: "text", text: "Second response" },
          ],
        });
      });

      it("converts assistant message with tool_use blocks", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
          messages: [
            { role: "user", content: "What is the weather?" },
            {
              role: "assistant",
              content: [
                {
                  type: "tool_use",
                  id: "toolu_123",
                  name: "get_weather",
                  input: { location: "San Francisco" },
                },
              ],
            },
          ],
        };

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages[1]).toEqual({
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "toolu_123",
              toolName: "get_weather",
              input: { location: "San Francisco" },
            },
          ],
        });
      });

      it("converts assistant message with mixed text and tool_use", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
          messages: [
            { role: "user", content: "What is the weather?" },
            {
              role: "assistant",
              content: [
                { type: "text", text: "Let me check that for you." },
                {
                  type: "tool_use",
                  id: "toolu_123",
                  name: "get_weather",
                  input: { location: "SF" },
                },
              ],
            },
          ],
        };

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages[1]).toEqual({
          role: "assistant",
          content: [
            { type: "text", text: "Let me check that for you." },
            {
              type: "tool-call",
              toolCallId: "toolu_123",
              toolName: "get_weather",
              input: { location: "SF" },
            },
          ],
        });
      });

      it("converts thinking blocks to text", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
          messages: [
            { role: "user", content: "Think about this" },
            {
              role: "assistant",
              content: [
                {
                  type: "thinking",
                  thinking: "Let me consider this carefully...",
                },
                { type: "text", text: "Here is my response." },
              ],
            },
          ],
        };

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages[1]).toEqual({
          role: "assistant",
          content: [
            { type: "text", text: "Let me consider this carefully..." },
            { type: "text", text: "Here is my response." },
          ],
        });
      });
    });

    describe("tool results", () => {
      it("converts tool_result in user message to tool message", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
          messages: [
            { role: "user", content: "What is the weather?" },
            {
              role: "assistant",
              content: [
                {
                  type: "tool_use",
                  id: "toolu_123",
                  name: "get_weather",
                  input: { location: "SF" },
                },
              ],
            },
            {
              role: "user",
              content: [
                {
                  type: "tool_result",
                  tool_use_id: "toolu_123",
                  content: "Sunny, 72°F",
                },
              ],
            },
          ],
        };

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages).toHaveLength(3);
        expect(result.messages[2]).toEqual({
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "toolu_123",
              toolName: "",
              output: { type: "text", value: "Sunny, 72°F" },
            },
          ],
        });
      });

      it("converts mixed tool_result and text in user message", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
          messages: [
            { role: "user", content: "What is the weather?" },
            {
              role: "assistant",
              content: [
                {
                  type: "tool_use",
                  id: "toolu_123",
                  name: "get_weather",
                  input: { location: "SF" },
                },
              ],
            },
            {
              role: "user",
              content: [
                {
                  type: "tool_result",
                  tool_use_id: "toolu_123",
                  content: "Sunny, 72°F",
                },
                { type: "text", text: "Thanks! What about tomorrow?" },
              ],
            },
          ],
        };

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages).toHaveLength(4);
        expect(result.messages[2]).toEqual({
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "toolu_123",
              toolName: "",
              output: { type: "text", value: "Sunny, 72°F" },
            },
          ],
        });
        expect(result.messages[3]).toEqual({
          role: "user",
          content: [{ type: "text", text: "Thanks! What about tomorrow?" }],
        });
      });

      it("converts tool_result with array content", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
          messages: [
            { role: "user", content: "Hello" },
            {
              role: "assistant",
              content: [
                {
                  type: "tool_use",
                  id: "toolu_123",
                  name: "search",
                  input: { query: "test" },
                },
              ],
            },
            {
              role: "user",
              content: [
                {
                  type: "tool_result",
                  tool_use_id: "toolu_123",
                  content: [
                    { type: "text", text: "Result 1" },
                    { type: "text", text: "Result 2" },
                  ],
                },
              ],
            },
          ],
        };

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages[2]).toEqual({
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "toolu_123",
              toolName: "",
              output: { type: "text", value: "Result 1\nResult 2" },
            },
          ],
        });
      });

      it("converts tool_result with empty content", () => {
        const input: MessageCreateParams = {
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 1024,
          messages: [
            { role: "user", content: "Hello" },
            {
              role: "assistant",
              content: [
                {
                  type: "tool_use",
                  id: "toolu_123",
                  name: "void_tool",
                  input: {},
                },
              ],
            },
            {
              role: "user",
              content: [
                {
                  type: "tool_result",
                  tool_use_id: "toolu_123",
                },
              ],
            },
          ],
        };

        const result = anthropicMessagesToAiSDK(input);

        expect(result.messages[2]).toEqual({
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "toolu_123",
              toolName: "",
              output: { type: "text", value: "" },
            },
          ],
        });
      });
    });
  });

  describe("tools", () => {
    it("converts tools with input_schema", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
        tools: [
          {
            name: "get_weather",
            description: "Get weather for a location",
            input_schema: {
              type: "object",
              properties: {
                location: { type: "string" },
              },
              required: ["location"],
            },
          },
        ],
      };

      const result = anthropicMessagesToAiSDK(input);

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
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
        tools: [
          {
            name: "tool_a",
            description: "Tool A",
            input_schema: { type: "object" },
          },
          {
            name: "tool_b",
            description: "Tool B",
            input_schema: { type: "object" },
          },
        ],
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(Object.keys(result.tools!)).toHaveLength(2);
      expect(result.tools!["tool_a"]).toBeDefined();
      expect(result.tools!["tool_b"]).toBeDefined();
    });

    it("returns undefined for empty tools array", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
        tools: [],
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.tools).toBeUndefined();
    });

    it("returns undefined when tools not provided", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.tools).toBeUndefined();
    });
  });

  describe("tool choice", () => {
    it("converts auto tool choice", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
        tool_choice: { type: "auto" },
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.toolChoice).toBe("auto");
    });

    it("converts any tool choice to required", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
        tool_choice: { type: "any" },
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.toolChoice).toBe("required");
    });

    it("converts none tool choice", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
        tool_choice: { type: "none" },
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.toolChoice).toBe("none");
    });

    it("converts specific tool choice", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
        tool_choice: { type: "tool", name: "get_weather" },
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.toolChoice).toEqual({
        type: "tool",
        toolName: "get_weather",
      });
    });

    it("returns undefined when tool_choice not provided", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.toolChoice).toBeUndefined();
    });
  });

  describe("full conversation", () => {
    it("converts complete multi-turn conversation with tools", () => {
      const input: MessageCreateParams = {
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        system: "You are a helpful weather assistant.",
        messages: [
          { role: "user", content: "What's the weather in San Francisco?" },
          {
            role: "assistant",
            content: [
              { type: "text", text: "I'll check that for you." },
              {
                type: "tool_use",
                id: "toolu_weather",
                name: "get_weather",
                input: { location: "San Francisco", unit: "fahrenheit" },
              },
            ],
          },
          {
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: "toolu_weather",
                content: '{"temperature": 72, "condition": "sunny"}',
              },
            ],
          },
          {
            role: "assistant",
            content: "It's sunny and 72°F in San Francisco!",
          },
        ],
        tools: [
          {
            name: "get_weather",
            description: "Get current weather",
            input_schema: {
              type: "object",
              properties: {
                location: { type: "string" },
                unit: { type: "string" },
              },
              required: ["location"],
            },
          },
        ],
        temperature: 0.7,
      };

      const result = anthropicMessagesToAiSDK(input);

      expect(result.model).toBe("claude-sonnet-4-5-20250929");
      expect(result.system).toBe("You are a helpful weather assistant.");
      expect(result.maxOutputTokens).toBe(1024);
      expect(result.temperature).toBe(0.7);
      expect(result.messages).toHaveLength(4);
      expect(result.tools).toBeDefined();
    });
  });
});
