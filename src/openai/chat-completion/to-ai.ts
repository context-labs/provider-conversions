import type {
  AssistantContent,
  ModelMessage,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  UserContent,
} from "ai";
import type { LanguageModelV2ToolResultOutput } from "@ai-sdk/provider";
import type { AiSDKParams } from "~/ai";
import type { ChatCompletionRequestBody } from "./types";
import type OpenAI from "openai";
import { safeJsonParse } from "~/utils";

type OpenAIMessage = OpenAI.ChatCompletionMessageParam;
type OpenAIUserMessage = OpenAI.ChatCompletionUserMessageParam;
type OpenAIAssistantMessage = OpenAI.ChatCompletionAssistantMessageParam;
type OpenAIToolMessage = OpenAI.ChatCompletionToolMessageParam;
type OpenAIFunctionMessage = OpenAI.ChatCompletionFunctionMessageParam;
type OpenAITool = OpenAI.ChatCompletionTool;
type OpenAIToolChoice = OpenAI.ChatCompletionToolChoiceOption;

export const openaiChatCompletionToAiSDK = (
  body: ChatCompletionRequestBody,
): AiSDKParams => {
  return {
    model: body.model,
    messages: convertMessages(body.messages),
    maxOutputTokens: body.max_completion_tokens ?? body.max_tokens ?? undefined,
    temperature: body.temperature ?? undefined,
    topP: body.top_p ?? undefined,
    frequencyPenalty: body.frequency_penalty ?? undefined,
    presencePenalty: body.presence_penalty ?? undefined,
    stopSequences: convertStopSequences(body.stop),
    seed: body.seed ?? undefined,
    tools: convertTools(body.tools),
    toolChoice: convertToolChoice(body.tool_choice),
  };
};

const convertMessages = (messages: OpenAIMessage[]): ModelMessage[] => {
  return messages.map((msg): ModelMessage => {
    switch (msg.role) {
      case "system":
      case "developer":
        return {
          role: "system",
          content: typeof msg.content === "string" ? msg.content : "",
        };

      case "user":
        return {
          role: "user",
          content: convertUserContent(msg as OpenAIUserMessage),
        };

      case "assistant":
        return {
          role: "assistant",
          content: convertAssistantContent(msg as OpenAIAssistantMessage),
        };

      case "tool":
        return convertToolMessage(msg as OpenAIToolMessage);

      case "function":
        return convertFunctionMessage(msg as OpenAIFunctionMessage);

      default: {
        const _exhaustive: never = msg;
        throw new Error(
          `Unknown message role: ${(_exhaustive as { role: string }).role}`,
        );
      }
    }
  });
};

const convertUserContent = (msg: OpenAIUserMessage): UserContent => {
  const { content } = msg;

  if (typeof content === "string") {
    return content;
  }

  return content.map((part) => {
    switch (part.type) {
      case "text":
        return { type: "text" as const, text: part.text };

      case "image_url":
        return {
          type: "image" as const,
          image: part.image_url.url,
        };

      case "input_audio":
        // Audio input not directly supported in AI SDK UserContent
        // Convert to empty text as fallback
        return { type: "text" as const, text: "" };

      case "file":
        // File input - convert to file part if we have file_data
        if (part.file.file_data) {
          return {
            type: "file" as const,
            data: part.file.file_data,
            mediaType: "application/octet-stream",
          };
        }
        // If only file_id, we can't convert directly
        return { type: "text" as const, text: "" };

      default: {
        const _exhaustive: never = part;
        throw new Error(
          `Unknown content part type: ${(_exhaustive as { type: string }).type}`,
        );
      }
    }
  });
};

const convertAssistantContent = (
  msg: OpenAIAssistantMessage,
): AssistantContent => {
  const parts: Array<TextPart | ToolCallPart> = [];

  // Add text content if present
  if (msg.content) {
    if (typeof msg.content === "string") {
      parts.push({ type: "text", text: msg.content });
    } else {
      for (const part of msg.content) {
        if (part.type === "text") {
          parts.push({ type: "text", text: part.text });
        }
        // Skip refusal parts - not directly mappable to AI SDK
      }
    }
  }

  // Add tool calls if present
  if (msg.tool_calls) {
    for (const toolCall of msg.tool_calls) {
      if (toolCall.type === "function") {
        parts.push({
          type: "tool-call",
          toolCallId: toolCall.id,
          toolName: toolCall.function.name,
          input: safeJsonParse(toolCall.function.arguments),
        });
      }
      // Skip custom tool calls - not directly mappable to AI SDK
    }
  }

  return parts;
};

const convertToolMessage = (msg: OpenAIToolMessage): ModelMessage => {
  return {
    role: "tool",
    content: [
      {
        type: "tool-result",
        toolCallId: msg.tool_call_id,
        toolName: "", // OpenAI doesn't include tool name in tool result messages
        output: convertToolResultOutput(msg.content),
      } satisfies ToolResultPart,
    ],
  };
};

const convertFunctionMessage = (msg: OpenAIFunctionMessage): ModelMessage => {
  // Deprecated, but handle for compatibility
  return {
    role: "tool",
    content: [
      {
        type: "tool-result",
        toolCallId: msg.name ?? "function",
        toolName: msg.name ?? "",
        output: convertToolResultOutput(msg.content),
      } satisfies ToolResultPart,
    ],
  };
};

const convertToolResultOutput = (
  content: string | Array<{ type: "text"; text: string }> | null,
): LanguageModelV2ToolResultOutput => {
  if (!content) {
    return { type: "text", value: "" };
  }

  if (typeof content === "string") {
    return { type: "text", value: content };
  }

  // Array of text parts - concatenate them
  const text = content.map((part) => part.text).join("");
  return { type: "text", value: text };
};

const convertStopSequences = (
  stop: ChatCompletionRequestBody["stop"],
): string[] | undefined => {
  if (stop === null || stop === undefined) return undefined;
  if (typeof stop === "string") return [stop];
  return stop;
};

const convertTools = (
  tools: OpenAITool[] | undefined,
): AiSDKParams["tools"] => {
  if (!tools || tools.length === 0) return undefined;

  const converted: NonNullable<AiSDKParams["tools"]> = {};

  for (const tool of tools) {
    if (tool.type === "function") {
      converted[tool.function.name] = {
        description: tool.function.description,
        inputSchema: tool.function.parameters ?? {},
      };
    }
    // Skip custom tools - not directly mappable to AI SDK
  }

  return Object.keys(converted).length > 0 ? converted : undefined;
};

const convertToolChoice = (
  toolChoice: OpenAIToolChoice | undefined,
): AiSDKParams["toolChoice"] => {
  if (!toolChoice) return undefined;

  // String literals
  if (toolChoice === "none") return "none";
  if (toolChoice === "auto") return "auto";
  if (toolChoice === "required") return "required";

  // Object types
  if (typeof toolChoice === "object") {
    // Named function tool choice: { type: "function", function: { name: "..." } }
    if ("type" in toolChoice && toolChoice.type === "function") {
      return {
        type: "tool",
        toolName: toolChoice.function.name,
      };
    }

    // Named custom tool choice: { type: "custom", custom: { name: "..." } }
    if ("type" in toolChoice && toolChoice.type === "custom") {
      return {
        type: "tool",
        toolName: toolChoice.custom.name,
      };
    }

    // Allowed tools choice: { type: "allowed_tools", allowed_tools: [...] }
    // Not directly mappable to AI SDK - return auto as fallback
    if ("type" in toolChoice && toolChoice.type === "allowed_tools") {
      return "auto";
    }
  }

  return undefined;
};

