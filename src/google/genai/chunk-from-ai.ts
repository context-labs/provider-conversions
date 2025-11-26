import type { AiSDKChunk } from "~/ai";
import type { GenerateContentResponse } from "./types";
import type { ToolSet } from "ai";
import type * as GenAI from "@google/genai";

type GooglePart = GenAI.Part;
type GoogleContent = GenAI.Content;

export interface ChunkConversionContext {
  /** The response ID to use for all chunks */
  responseId: string;
  /** The model version to use */
  modelVersion: string;
  /** Accumulated text content */
  accumulatedText: string;
  /** Accumulated function calls */
  accumulatedFunctionCalls: Map<string, { name: string; args: string }>;
  /** Track input tokens */
  inputTokens: number;
  /** Track output tokens */
  outputTokens: number;
}

export const createChunkConversionContext = (options: {
  responseId: string;
  modelVersion: string;
}): ChunkConversionContext => ({
  responseId: options.responseId,
  modelVersion: options.modelVersion,
  accumulatedText: "",
  accumulatedFunctionCalls: new Map(),
  inputTokens: 0,
  outputTokens: 0,
});

export type ChunkConversionResult =
  | { type: "response"; response: GenerateContentResponse }
  | { type: "skip" }
  | { type: "error"; error: string };

/**
 * Convert an AI SDK stream chunk to a Google GenAI GenerateContentResponse.
 *
 * Google GenAI streaming returns incremental GenerateContentResponse objects,
 * so each chunk is converted to a full response with accumulated content.
 * The context object is mutated to track state across chunks.
 */
export const googleChunkFromAiSDK = <T extends ToolSet>(
  chunk: AiSDKChunk<T>,
  context: ChunkConversionContext,
): ChunkConversionResult => {
  switch (chunk.type) {
    case "text-start":
      // Initialize text accumulation
      return { type: "skip" };

    case "text-delta": {
      context.accumulatedText += chunk.text;
      return {
        type: "response",
        response: buildIncrementalResponse(context),
      };
    }

    case "text-end":
      // No action needed, text is already accumulated
      return { type: "skip" };

    case "reasoning-start":
    case "reasoning-delta":
    case "reasoning-end":
      // Google doesn't have a separate reasoning stream in the same way
      // We could map to thought parts, but skip for now
      return { type: "skip" };

    case "tool-input-start": {
      // Initialize function call accumulation
      context.accumulatedFunctionCalls.set(chunk.id, {
        name: chunk.toolName,
        args: "",
      });
      return { type: "skip" };
    }

    case "tool-input-delta": {
      const funcCall = context.accumulatedFunctionCalls.get(chunk.id);
      if (!funcCall) {
        return { type: "error", error: `Unknown tool call id: ${chunk.id}` };
      }
      funcCall.args += chunk.delta;

      return {
        type: "response",
        response: buildIncrementalResponse(context),
      };
    }

    case "tool-input-end": {
      // Function call is complete, emit final response with parsed args
      return {
        type: "response",
        response: buildIncrementalResponse(context),
      };
    }

    case "tool-call": {
      // Complete tool call (non-streaming)
      context.accumulatedFunctionCalls.set(chunk.toolCallId, {
        name: chunk.toolName,
        args: JSON.stringify(chunk.input),
      });

      return {
        type: "response",
        response: buildIncrementalResponse(context),
      };
    }

    case "tool-result":
    case "tool-error":
      // Tool results are not part of the model's stream response
      return { type: "skip" };

    case "finish-step": {
      // Update usage from step finish
      if (chunk.usage.inputTokens !== undefined) {
        context.inputTokens = chunk.usage.inputTokens;
      }
      if (chunk.usage.outputTokens !== undefined) {
        context.outputTokens = chunk.usage.outputTokens;
      }
      return { type: "skip" };
    }

    case "finish": {
      // Final response with finish reason and usage
      return {
        type: "response",
        response: buildFinalResponse(context, chunk.finishReason, chunk.totalUsage),
      };
    }

    case "start":
    case "start-step":
      return { type: "skip" };

    case "abort":
      return { type: "skip" };

    case "error":
      return { type: "error", error: String(chunk.error) };

    case "source":
    case "file":
    case "raw":
      // No direct equivalents in Google GenAI
      return { type: "skip" };

    default: {
      // Exhaustive check
      const _exhaustive: never = chunk;
      return {
        type: "error",
        error: `Unknown chunk type: ${(_exhaustive as { type: string }).type}`,
      };
    }
  }
};

const buildIncrementalResponse = (
  context: ChunkConversionContext,
): GenerateContentResponse => {
  const content = buildContent(context);

  return {
    responseId: context.responseId,
    modelVersion: context.modelVersion,
    candidates: [
      {
        content,
        finishReason: undefined,
      },
    ],
    usageMetadata: {
      promptTokenCount: context.inputTokens,
      candidatesTokenCount: context.outputTokens,
      totalTokenCount: context.inputTokens + context.outputTokens,
    } as GenAI.GenerateContentResponseUsageMetadata,
  } as GenerateContentResponse;
};

const buildFinalResponse = (
  context: ChunkConversionContext,
  finishReason: string,
  totalUsage: { inputTokens?: number; outputTokens?: number; totalTokens?: number },
): GenerateContentResponse => {
  const content = buildContent(context);
  const inputTokens = totalUsage.inputTokens ?? context.inputTokens;
  const outputTokens = totalUsage.outputTokens ?? context.outputTokens;

  return {
    responseId: context.responseId,
    modelVersion: context.modelVersion,
    candidates: [
      {
        content,
        finishReason: convertFinishReason(finishReason),
      },
    ],
    usageMetadata: {
      promptTokenCount: inputTokens,
      candidatesTokenCount: outputTokens,
      totalTokenCount: totalUsage.totalTokens ?? inputTokens + outputTokens,
    } as GenAI.GenerateContentResponseUsageMetadata,
  } as GenerateContentResponse;
};

const buildContent = (context: ChunkConversionContext): GoogleContent => {
  const parts: GooglePart[] = [];

  // Add accumulated text
  if (context.accumulatedText) {
    parts.push({ text: context.accumulatedText });
  }

  // Add accumulated function calls
  for (const [id, funcCall] of context.accumulatedFunctionCalls) {
    let args: Record<string, unknown> = {};
    try {
      if (funcCall.args) {
        args = JSON.parse(funcCall.args) as Record<string, unknown>;
      }
    } catch {
      // If JSON parse fails, use empty object
      args = {};
    }

    parts.push({
      functionCall: {
        id,
        name: funcCall.name,
        args,
      },
    });
  }

  // If no parts, add empty text
  if (parts.length === 0) {
    parts.push({ text: "" });
  }

  return {
    role: "model",
    parts,
  };
};

const convertFinishReason = (finishReason: string): GenAI.FinishReason => {
  switch (finishReason) {
    case "stop":
      return "STOP" as GenAI.FinishReason;
    case "length":
      return "MAX_TOKENS" as GenAI.FinishReason;
    case "content-filter":
      return "SAFETY" as GenAI.FinishReason;
    case "tool-calls":
      return "STOP" as GenAI.FinishReason;
    default:
      return "OTHER" as GenAI.FinishReason;
  }
};
