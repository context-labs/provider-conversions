import type { AiSDKResponse } from "~/ai";
import type { ChatCompletionResponse } from "./types";
import type OpenAI from "openai";

type OpenAIFinishReason = NonNullable<
  OpenAI.ChatCompletion["choices"][number]["finish_reason"]
>;
type OpenAIToolCall = OpenAI.ChatCompletionMessageToolCall;
type OpenAIUsage = OpenAI.CompletionUsage;

export interface FromAiSDKOptions {
  /** Override the response ID. If not provided, uses the AI SDK response ID. */
  id?: string;
  /** Override the model name in the response. */
  model?: string;
  /** Override the created timestamp (unix seconds). If not provided, uses the AI SDK response timestamp. */
  created?: number;
}

export const openaiChatCompletionFromAiSDK = (
  response: AiSDKResponse,
  options?: FromAiSDKOptions,
): ChatCompletionResponse => {
  const toolCalls = convertToolCalls(response.toolCalls);
  const hasToolCalls = toolCalls.length > 0;

  return {
    id: options?.id ?? response.response.id,
    object: "chat.completion",
    created:
      options?.created ?? Math.floor(response.response.timestamp.getTime() / 1000),
    model: options?.model ?? response.response.modelId,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: response.text || null,
          refusal: null,
          ...(hasToolCalls && { tool_calls: toolCalls }),
        },
        logprobs: null,
        finish_reason: convertFinishReason(response.finishReason),
      },
    ],
    usage: convertUsage(response.usage),
  };
};

const convertFinishReason = (
  finishReason: AiSDKResponse["finishReason"],
): OpenAIFinishReason => {
  switch (finishReason) {
    case "stop":
      return "stop";
    case "length":
      return "length";
    case "content-filter":
      return "content_filter";
    case "tool-calls":
      return "tool_calls";
    case "error":
    case "other":
    case "unknown":
    default:
      // OpenAI doesn't have direct equivalents for these
      // Map to 'stop' as the safest default
      return "stop";
  }
};

const convertToolCalls = (
  toolCalls: AiSDKResponse["toolCalls"],
): OpenAIToolCall[] => {
  return toolCalls.map(
    (toolCall): OpenAIToolCall => ({
      id: toolCall.toolCallId,
      type: "function",
      function: {
        name: toolCall.toolName,
        arguments: JSON.stringify(toolCall.input),
      },
    }),
  );
};

const convertUsage = (
  usage: AiSDKResponse["usage"],
): OpenAIUsage => {
  return {
    prompt_tokens: usage.inputTokens ?? 0,
    completion_tokens: usage.outputTokens ?? 0,
    total_tokens: usage.totalTokens ?? (usage.inputTokens ?? 0) + (usage.outputTokens ?? 0),
  };
};
