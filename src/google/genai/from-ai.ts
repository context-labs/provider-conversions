import type { AiSDKResponse } from "~/ai";
import type { GenerateContentResponse } from "./types";
import type * as GenAI from "@google/genai";

type GooglePart = GenAI.Part;
type GoogleContent = GenAI.Content;
type GoogleCandidate = GenAI.Candidate;

export interface FromAiSDKOptions {
  /** Override the response ID. If not provided, uses the AI SDK response ID. */
  responseId?: string;
  /** Override the model version. If not provided, uses the AI SDK model ID. */
  modelVersion?: string;
}

export const googleGenaiFromAiSDK = (
  response: AiSDKResponse,
  options?: FromAiSDKOptions,
): GenerateContentResponse => {
  const content = buildContent(response);
  const candidate = buildCandidate(response, content);

  // GenerateContentResponse is a class in the SDK, but we can create
  // a plain object that matches its shape
  return {
    responseId: options?.responseId ?? response.response.id,
    modelVersion: options?.modelVersion ?? response.response.modelId,
    candidates: [candidate],
    usageMetadata: buildUsageMetadata(response),
  } as GenerateContentResponse;
};

const buildContent = (response: AiSDKResponse): GoogleContent => {
  const parts: GooglePart[] = [];

  // Add text content if present
  if (response.text) {
    parts.push({ text: response.text });
  }

  // Add tool calls as function calls
  for (const toolCall of response.toolCalls) {
    parts.push({
      functionCall: {
        id: toolCall.toolCallId,
        name: toolCall.toolName,
        args: toolCall.input as Record<string, unknown>,
      },
    });
  }

  return {
    role: "model",
    parts: parts.length > 0 ? parts : [{ text: "" }],
  };
};

const buildCandidate = (
  response: AiSDKResponse,
  content: GoogleContent,
): GoogleCandidate => {
  return {
    content,
    finishReason: convertFinishReason(response.finishReason),
  };
};

const convertFinishReason = (
  finishReason: AiSDKResponse["finishReason"],
): GenAI.FinishReason => {
  switch (finishReason) {
    case "stop":
      return "STOP" as GenAI.FinishReason;
    case "length":
      return "MAX_TOKENS" as GenAI.FinishReason;
    case "content-filter":
      return "SAFETY" as GenAI.FinishReason;
    case "tool-calls":
      // Google doesn't have a separate finish reason for tool calls
      // The model stops and the content contains function calls
      return "STOP" as GenAI.FinishReason;
    case "error":
    case "other":
    case "unknown":
    default:
      return "OTHER" as GenAI.FinishReason;
  }
};

const buildUsageMetadata = (
  response: AiSDKResponse,
): GenAI.GenerateContentResponseUsageMetadata => {
  const inputTokens = response.usage.inputTokens ?? 0;
  const outputTokens = response.usage.outputTokens ?? 0;

  return {
    promptTokenCount: inputTokens,
    candidatesTokenCount: outputTokens,
    totalTokenCount: response.usage.totalTokens ?? inputTokens + outputTokens,
  } as GenAI.GenerateContentResponseUsageMetadata;
};
