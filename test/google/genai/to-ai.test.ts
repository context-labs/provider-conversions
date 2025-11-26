import { describe, it, expect } from "vitest";
import { googleGenaiToAiSDK } from "~/google/genai/to-ai";
import type { GenerateContentParameters } from "~/google/genai/types";

describe("googleGenaiToAiSDK", () => {
  describe("model", () => {
    it("passes through model name", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.model).toBe("gemini-2.0-flash");
    });
  });

  describe("system instruction", () => {
    it("converts string system instruction", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: {
          systemInstruction: "You are a helpful assistant.",
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.system).toBe("You are a helpful assistant.");
    });

    it("converts Content object system instruction", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: {
          systemInstruction: {
            role: "user",
            parts: [{ text: "Be concise." }, { text: "Be helpful." }],
          },
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.system).toBe("Be concise.\nBe helpful.");
    });

    it("converts Part array system instruction", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: {
          systemInstruction: [{ text: "Part 1" }, { text: "Part 2" }],
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.system).toBe("Part 1\nPart 2");
    });

    it("converts single Part system instruction", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: {
          systemInstruction: { text: "Single part instruction" },
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.system).toBe("Single part instruction");
    });

    it("returns undefined when no system instruction", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.system).toBeUndefined();
    });
  });

  describe("generation config", () => {
    it("converts temperature", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: { temperature: 0.7 },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.temperature).toBe(0.7);
    });

    it("converts topP", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: { topP: 0.9 },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.topP).toBe(0.9);
    });

    it("converts topK", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: { topK: 40 },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.topK).toBe(40);
    });

    it("converts maxOutputTokens", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: { maxOutputTokens: 1000 },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.maxOutputTokens).toBe(1000);
    });

    it("converts stopSequences", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: { stopSequences: ["END", "STOP"] },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.stopSequences).toEqual(["END", "STOP"]);
    });

    it("converts presencePenalty", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: { presencePenalty: 0.5 },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.presencePenalty).toBe(0.5);
    });

    it("converts frequencyPenalty", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: { frequencyPenalty: 0.3 },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.frequencyPenalty).toBe(0.3);
    });

    it("converts seed", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: { seed: 12345 },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.seed).toBe(12345);
    });
  });

  describe("user messages", () => {
    it("converts simple text message", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello, world!" }] }],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages).toHaveLength(1);
      expect(result.messages![0]).toEqual({
        role: "user",
        content: [{ type: "text", text: "Hello, world!" }],
      });
    });

    it("converts multiple text parts", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [
          {
            role: "user",
            parts: [{ text: "First part" }, { text: "Second part" }],
          },
        ],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages).toHaveLength(1);
      expect(result.messages![0]).toEqual({
        role: "user",
        content: [
          { type: "text", text: "First part" },
          { type: "text", text: "Second part" },
        ],
      });
    });

    it("converts inline image data", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [
          {
            role: "user",
            parts: [
              {
                inlineData: {
                  mimeType: "image/png",
                  data: "base64encodeddata",
                },
              },
            ],
          },
        ],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages).toHaveLength(1);
      expect(result.messages![0]).toEqual({
        role: "user",
        content: [
          { type: "image", image: "data:image/png;base64,base64encodeddata" },
        ],
      });
    });

    it("converts inline file data", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [
          {
            role: "user",
            parts: [
              {
                inlineData: {
                  mimeType: "application/pdf",
                  data: "pdfbase64data",
                },
              },
            ],
          },
        ],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages).toHaveLength(1);
      expect(result.messages![0]).toEqual({
        role: "user",
        content: [
          { type: "file", data: "pdfbase64data", mediaType: "application/pdf" },
        ],
      });
    });

    it("converts file data URI reference to text", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [
          {
            role: "user",
            parts: [
              {
                fileData: {
                  fileUri: "gs://bucket/file.pdf",
                  mimeType: "application/pdf",
                },
              },
            ],
          },
        ],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages).toHaveLength(1);
      expect(result.messages![0]).toEqual({
        role: "user",
        content: [{ type: "text", text: "[File: gs://bucket/file.pdf]" }],
      });
    });

    it("handles empty user content", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [] }],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages).toHaveLength(1);
      expect(result.messages![0]).toEqual({
        role: "user",
        content: "",
      });
    });
  });

  describe("model messages", () => {
    it("converts model role to assistant", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [
          { role: "user", parts: [{ text: "Hello" }] },
          { role: "model", parts: [{ text: "Hi there!" }] },
        ],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages).toHaveLength(2);
      expect(result.messages![1]).toEqual({
        role: "assistant",
        content: [{ type: "text", text: "Hi there!" }],
      });
    });

    it("converts model function call", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [
          { role: "user", parts: [{ text: "What's the weather?" }] },
          {
            role: "model",
            parts: [
              {
                functionCall: {
                  id: "call_123",
                  name: "get_weather",
                  args: { location: "San Francisco" },
                },
              },
            ],
          },
        ],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages).toHaveLength(2);
      expect(result.messages![1]).toEqual({
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

    it("converts model text with function call", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [
          { role: "user", parts: [{ text: "What's the weather?" }] },
          {
            role: "model",
            parts: [
              { text: "Let me check that for you." },
              {
                functionCall: {
                  name: "get_weather",
                  args: { location: "NYC" },
                },
              },
            ],
          },
        ],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages![1]).toEqual({
        role: "assistant",
        content: [
          { type: "text", text: "Let me check that for you." },
          {
            type: "tool-call",
            toolCallId: "get_weather",
            toolName: "get_weather",
            input: { location: "NYC" },
          },
        ],
      });
    });

    it("handles thought parts", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [
          { role: "user", parts: [{ text: "Think about this" }] },
          {
            role: "model",
            parts: [
              { text: "Let me think...", thought: true },
              { text: "Here's my answer." },
            ],
          },
        ],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages![1]).toEqual({
        role: "assistant",
        content: [
          { type: "text", text: "Let me think..." },
          { type: "text", text: "Here's my answer." },
        ],
      });
    });
  });

  describe("function responses (tool results)", () => {
    it("converts function response to tool result", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [
          {
            role: "user",
            parts: [
              {
                functionResponse: {
                  id: "call_123",
                  name: "get_weather",
                  response: { output: "Sunny, 72°F" },
                },
              },
            ],
          },
        ],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages).toHaveLength(1);
      expect(result.messages![0]).toEqual({
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "call_123",
            toolName: "get_weather",
            output: { type: "text", value: "Sunny, 72°F" },
          },
        ],
      });
    });

    it("converts function response with error", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [
          {
            role: "user",
            parts: [
              {
                functionResponse: {
                  name: "failing_tool",
                  response: { error: "Connection failed" },
                },
              },
            ],
          },
        ],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages![0]).toEqual({
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "failing_tool",
            toolName: "failing_tool",
            output: { type: "text", value: "Connection failed" },
          },
        ],
      });
    });

    it("stringifies complex response objects", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [
          {
            role: "user",
            parts: [
              {
                functionResponse: {
                  id: "call_456",
                  name: "search",
                  response: { results: [{ title: "Result 1" }], count: 1 },
                },
              },
            ],
          },
        ],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages![0]).toEqual({
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "call_456",
            toolName: "search",
            output: {
              type: "text",
              value: JSON.stringify({ results: [{ title: "Result 1" }], count: 1 }),
            },
          },
        ],
      });
    });

    it("handles mixed user content and function response", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [
          {
            role: "user",
            parts: [
              {
                functionResponse: {
                  id: "call_789",
                  name: "get_data",
                  response: { output: "data result" },
                },
              },
              { text: "Now summarize this." },
            ],
          },
        ],
      };

      const result = googleGenaiToAiSDK(params);

      // Tool result comes first, then user content
      expect(result.messages).toHaveLength(2);
      expect(result.messages![0]).toEqual({
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "call_789",
            toolName: "get_data",
            output: { type: "text", value: "data result" },
          },
        ],
      });
      expect(result.messages![1]).toEqual({
        role: "user",
        content: [{ type: "text", text: "Now summarize this." }],
      });
    });
  });

  describe("tools", () => {
    it("converts function declarations to tools", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: {
          tools: [
            {
              functionDeclarations: [
                {
                  name: "get_weather",
                  description: "Get the weather for a location",
                  parameters: {
                    type: "OBJECT",
                    properties: {
                      location: {
                        type: "STRING",
                        description: "The city name",
                      },
                    },
                    required: ["location"],
                  },
                },
              ],
            },
          ],
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.tools).toEqual({
        get_weather: {
          description: "Get the weather for a location",
          inputSchema: {
            type: "object",
            properties: {
              location: {
                type: "string",
                description: "The city name",
              },
            },
            required: ["location"],
          },
        },
      });
    });

    it("converts multiple function declarations", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: {
          tools: [
            {
              functionDeclarations: [
                { name: "tool_a", description: "Tool A" },
                { name: "tool_b", description: "Tool B" },
              ],
            },
          ],
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.tools).toEqual({
        tool_a: {
          description: "Tool A",
          inputSchema: { type: "object", properties: {} },
        },
        tool_b: {
          description: "Tool B",
          inputSchema: { type: "object", properties: {} },
        },
      });
    });

    it("uses parametersJsonSchema when available", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: {
          tools: [
            {
              functionDeclarations: [
                {
                  name: "custom_tool",
                  parametersJsonSchema: {
                    type: "object",
                    properties: {
                      input: { type: "string" },
                    },
                  },
                },
              ],
            },
          ],
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.tools).toEqual({
        custom_tool: {
          description: undefined,
          inputSchema: {
            type: "object",
            properties: { input: { type: "string" } },
          },
        },
      });
    });

    it("returns undefined when no tools", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.tools).toBeUndefined();
    });
  });

  describe("tool choice", () => {
    it("converts AUTO mode to auto", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: {
          toolConfig: {
            functionCallingConfig: { mode: "AUTO" },
          },
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.toolChoice).toBe("auto");
    });

    it("converts MODE_UNSPECIFIED to auto", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: {
          toolConfig: {
            functionCallingConfig: { mode: "MODE_UNSPECIFIED" },
          },
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.toolChoice).toBe("auto");
    });

    it("converts VALIDATED to auto", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: {
          toolConfig: {
            functionCallingConfig: { mode: "VALIDATED" },
          },
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.toolChoice).toBe("auto");
    });

    it("converts ANY mode to required", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: {
          toolConfig: {
            functionCallingConfig: { mode: "ANY" },
          },
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.toolChoice).toBe("required");
    });

    it("converts ANY with single allowed function to tool choice", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: {
          toolConfig: {
            functionCallingConfig: {
              mode: "ANY",
              allowedFunctionNames: ["specific_tool"],
            },
          },
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.toolChoice).toEqual({
        type: "tool",
        toolName: "specific_tool",
      });
    });

    it("converts ANY with multiple allowed functions to required", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: {
          toolConfig: {
            functionCallingConfig: {
              mode: "ANY",
              allowedFunctionNames: ["tool_a", "tool_b"],
            },
          },
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.toolChoice).toBe("required");
    });

    it("converts NONE mode to none", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
        config: {
          toolConfig: {
            functionCallingConfig: { mode: "NONE" },
          },
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.toolChoice).toBe("none");
    });

    it("returns undefined when no tool config", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ role: "user", parts: [{ text: "Hello" }] }],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.toolChoice).toBeUndefined();
    });
  });

  describe("content list union handling", () => {
    it("handles Content array", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [
          { role: "user", parts: [{ text: "First" }] },
          { role: "model", parts: [{ text: "Response" }] },
          { role: "user", parts: [{ text: "Second" }] },
        ],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages).toHaveLength(3);
    });

    it("handles single Content object", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: { role: "user", parts: [{ text: "Single content" }] },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages).toHaveLength(1);
      expect(result.messages![0]).toEqual({
        role: "user",
        content: [{ type: "text", text: "Single content" }],
      });
    });

    it("handles Part array (wraps in user content)", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [{ text: "Part 1" }, { text: "Part 2" }],
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages).toHaveLength(1);
      expect(result.messages![0]).toEqual({
        role: "user",
        content: [
          { type: "text", text: "Part 1" },
          { type: "text", text: "Part 2" },
        ],
      });
    });

    it("handles single Part (wraps in user content)", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: { text: "Single part" },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages).toHaveLength(1);
      expect(result.messages![0]).toEqual({
        role: "user",
        content: [{ type: "text", text: "Single part" }],
      });
    });

    it("handles string content (wraps in user content)", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: "Just a string",
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.messages).toHaveLength(1);
      expect(result.messages![0]).toEqual({
        role: "user",
        content: [{ type: "text", text: "Just a string" }],
      });
    });
  });

  describe("full conversation", () => {
    it("converts a complete multi-turn conversation with tool use", () => {
      const params: GenerateContentParameters = {
        model: "gemini-2.0-flash",
        contents: [
          { role: "user", parts: [{ text: "What's the weather in SF?" }] },
          {
            role: "model",
            parts: [
              { text: "Let me check." },
              {
                functionCall: {
                  id: "call_weather",
                  name: "get_weather",
                  args: { location: "San Francisco" },
                },
              },
            ],
          },
          {
            role: "user",
            parts: [
              {
                functionResponse: {
                  id: "call_weather",
                  name: "get_weather",
                  response: { output: "Sunny, 68°F" },
                },
              },
            ],
          },
          {
            role: "model",
            parts: [{ text: "The weather in San Francisco is sunny and 68°F." }],
          },
        ],
        config: {
          systemInstruction: "You are a helpful weather assistant.",
          temperature: 0.5,
          maxOutputTokens: 500,
          tools: [
            {
              functionDeclarations: [
                {
                  name: "get_weather",
                  description: "Get weather for a location",
                  parameters: {
                    type: "OBJECT",
                    properties: {
                      location: { type: "STRING" },
                    },
                  },
                },
              ],
            },
          ],
          toolConfig: {
            functionCallingConfig: { mode: "AUTO" },
          },
        },
      };

      const result = googleGenaiToAiSDK(params);

      expect(result.model).toBe("gemini-2.0-flash");
      expect(result.system).toBe("You are a helpful weather assistant.");
      expect(result.temperature).toBe(0.5);
      expect(result.maxOutputTokens).toBe(500);
      expect(result.toolChoice).toBe("auto");
      expect(result.tools).toBeDefined();
      expect(result.messages).toHaveLength(4);

      // User message
      expect(result.messages![0].role).toBe("user");

      // Assistant with tool call
      expect(result.messages![1].role).toBe("assistant");

      // Tool result
      expect(result.messages![2].role).toBe("tool");

      // Final assistant response
      expect(result.messages![3].role).toBe("assistant");
    });
  });
});
