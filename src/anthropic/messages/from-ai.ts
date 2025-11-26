import type { AiSDKResponse } from "~/ai";
import type { Message } from "./types";
import type Anthropic from "@anthropic-ai/sdk";

type AnthropicContentBlock = Anthropic.ContentBlock;
type AnthropicTextBlock = Anthropic.TextBlock;
type AnthropicToolUseBlock = Anthropic.ToolUseBlock;
type AnthropicStopReason = Anthropic.Message["stop_reason"];
type AnthropicUsage = Anthropic.Usage;

export interface FromAiSDKOptions {
  /** Override the response ID. If not provided, uses the AI SDK response ID. */
  id?: string;
  /** Override the model name in the response. */
  model?: string;
}

export const anthropicMessagesFromAiSDK = (
  response: AiSDKResponse,
  options?: FromAiSDKOptions,
): Message => {
  const content = buildContentBlocks(response);

  return {
    id: options?.id ?? response.response.id,
    type: "message",
    role: "assistant",
    model: (options?.model ?? response.response.modelId) as Anthropic.Model,
    content,
    stop_reason: convertStopReason(response.finishReason),
    stop_sequence: null,
    usage: convertUsage(response.usage),
  };
};

const buildContentBlocks = (response: AiSDKResponse): AnthropicContentBlock[] => {
  const blocks: AnthropicContentBlock[] = [];

  // Add text content if present
  if (response.text) {
    blocks.push(createTextBlock(response.text));
  }

  // Add tool calls
  for (const toolCall of response.toolCalls) {
    blocks.push(createToolUseBlock(toolCall));
  }

  // If no content, add empty text block (Anthropic requires at least one content block)
  if (blocks.length === 0) {
    blocks.push(createTextBlock(""));
  }

  return blocks;
};

const createTextBlock = (text: string): AnthropicTextBlock => ({
  type: "text",
  text,
  citations: null,
});

const createToolUseBlock = (
  toolCall: AiSDKResponse["toolCalls"][number],
): AnthropicToolUseBlock => ({
  type: "tool_use",
  id: toolCall.toolCallId,
  name: toolCall.toolName,
  input: toolCall.input,
});

const convertStopReason = (
  finishReason: AiSDKResponse["finishReason"],
): AnthropicStopReason => {
  switch (finishReason) {
    case "stop":
      return "end_turn";
    case "length":
      return "max_tokens";
    case "tool-calls":
      return "tool_use";
    case "content-filter":
      return "refusal";
    case "error":
    case "other":
    case "unknown":
    default:
      return "end_turn";
  }
};

const convertUsage = (usage: AiSDKResponse["usage"]): AnthropicUsage => ({
  input_tokens: usage.inputTokens ?? 0,
  output_tokens: usage.outputTokens ?? 0,
  cache_creation_input_tokens: null,
  cache_read_input_tokens: null,
  cache_creation: null,
  server_tool_use: null,
  service_tier: null,
});
