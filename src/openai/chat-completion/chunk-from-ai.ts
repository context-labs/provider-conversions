import type { AiSDKChunk } from "~/ai";
import type { ChatCompletionChunk } from "./types";
import type { ToolSet } from "ai";
import type OpenAI from "openai";

type OpenAIChunkChoice = ChatCompletionChunk["choices"][number];
type OpenAIChunkDelta = OpenAIChunkChoice["delta"];
type OpenAIChunkFinishReason = OpenAIChunkChoice["finish_reason"];
type OpenAIChunkToolCall = NonNullable<OpenAIChunkDelta["tool_calls"]>[number];
type OpenAIUsage = OpenAI.CompletionUsage;

export interface ChunkConversionContext {
  /** The response ID to use for all chunks */
  id: string;
  /** The model name to use for all chunks */
  model: string;
  /** The created timestamp (unix seconds) to use for all chunks */
  created: number;
  /** Track tool call indices for streaming tool calls */
  toolCallIndices: Map<string, number>;
  /** Counter for assigning tool call indices */
  nextToolCallIndex: number;
}

export const createChunkConversionContext = (options: {
  id: string;
  model: string;
  created?: number;
}): ChunkConversionContext => ({
  id: options.id,
  model: options.model,
  created: options.created ?? Math.floor(Date.now() / 1000),
  toolCallIndices: new Map(),
  nextToolCallIndex: 0,
});

export type ChunkConversionResult =
  | { type: "chunk"; chunk: ChatCompletionChunk }
  | { type: "skip" }
  | { type: "error"; error: string };

/**
 * Convert an AI SDK stream chunk to an OpenAI ChatCompletionChunk.
 *
 * Some AI SDK chunk types don't have direct OpenAI equivalents and will return { type: "skip" }.
 * The context object is mutated to track state across chunks (e.g., tool call indices).
 */
export const openaiChunkFromAiSDK = <T extends ToolSet>(
  chunk: AiSDKChunk<T>,
  context: ChunkConversionContext,
): ChunkConversionResult => {
  switch (chunk.type) {
    case "text-delta":
      return {
        type: "chunk",
        chunk: createChunk(context, {
          delta: { content: chunk.text },
          finish_reason: null,
        }),
      };

    case "text-start":
      // OpenAI sends role in the first chunk
      return {
        type: "chunk",
        chunk: createChunk(context, {
          delta: { role: "assistant", content: "" },
          finish_reason: null,
        }),
      };

    case "text-end":
      // No direct equivalent in OpenAI streaming
      return { type: "skip" };

    case "reasoning-delta":
      // OpenAI doesn't have a direct equivalent for reasoning
      // Could map to content if needed, but skip for now
      return { type: "skip" };

    case "reasoning-start":
    case "reasoning-end":
      return { type: "skip" };

    case "tool-input-start": {
      // Get or assign an index for this tool call
      let index = context.toolCallIndices.get(chunk.id);
      if (index === undefined) {
        index = context.nextToolCallIndex++;
        context.toolCallIndices.set(chunk.id, index);
      }

      return {
        type: "chunk",
        chunk: createChunk(context, {
          delta: {
            tool_calls: [
              {
                index,
                id: chunk.id,
                type: "function",
                function: {
                  name: chunk.toolName,
                  arguments: "",
                },
              },
            ],
          },
          finish_reason: null,
        }),
      };
    }

    case "tool-input-delta": {
      const index = context.toolCallIndices.get(chunk.id);
      if (index === undefined) {
        return { type: "error", error: `Unknown tool call id: ${chunk.id}` };
      }

      return {
        type: "chunk",
        chunk: createChunk(context, {
          delta: {
            tool_calls: [
              {
                index,
                function: {
                  arguments: chunk.delta,
                },
              },
            ],
          },
          finish_reason: null,
        }),
      };
    }

    case "tool-input-end":
      // No direct equivalent - OpenAI doesn't send an explicit end
      return { type: "skip" };

    case "tool-call":
      // This is the complete tool call (non-streaming)
      // In streaming mode, we would have received tool-input-* events
      // But if we get this, emit it as a complete tool call
      {
        let index = context.toolCallIndices.get(chunk.toolCallId);
        if (index === undefined) {
          index = context.nextToolCallIndex++;
          context.toolCallIndices.set(chunk.toolCallId, index);
        }

        return {
          type: "chunk",
          chunk: createChunk(context, {
            delta: {
              tool_calls: [
                {
                  index,
                  id: chunk.toolCallId,
                  type: "function",
                  function: {
                    name: chunk.toolName,
                    arguments: JSON.stringify(chunk.input),
                  },
                },
              ],
            },
            finish_reason: null,
          }),
        };
      }

    case "tool-result":
    case "tool-error":
      // Tool results are not part of the assistant's stream response in OpenAI
      return { type: "skip" };

    case "finish-step":
      return {
        type: "chunk",
        chunk: createChunk(
          context,
          {
            delta: {},
            finish_reason: convertFinishReason(chunk.finishReason),
          },
          convertUsage(chunk.usage),
        ),
      };

    case "finish":
      return {
        type: "chunk",
        chunk: createChunk(
          context,
          {
            delta: {},
            finish_reason: convertFinishReason(chunk.finishReason),
          },
          convertUsage(chunk.totalUsage),
        ),
      };

    case "start":
    case "start-step":
      // Could emit an initial chunk with role, but text-start handles this
      return { type: "skip" };

    case "abort":
      // No direct equivalent
      return { type: "skip" };

    case "error":
      return { type: "error", error: String(chunk.error) };

    case "source":
    case "file":
    case "raw":
      // No direct equivalents in OpenAI
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

const createChunk = (
  context: ChunkConversionContext,
  choice: {
    delta: OpenAIChunkDelta;
    finish_reason: OpenAIChunkFinishReason;
  },
  usage?: OpenAIUsage | null,
): ChatCompletionChunk => ({
  id: context.id,
  object: "chat.completion.chunk",
  created: context.created,
  model: context.model,
  choices: [
    {
      index: 0,
      delta: choice.delta,
      finish_reason: choice.finish_reason,
      logprobs: null,
    },
  ],
  ...(usage !== undefined && { usage }),
});

const convertFinishReason = (
  finishReason: string,
): OpenAIChunkFinishReason => {
  switch (finishReason) {
    case "stop":
      return "stop";
    case "length":
      return "length";
    case "content-filter":
      return "content_filter";
    case "tool-calls":
      return "tool_calls";
    default:
      return "stop";
  }
};

const convertUsage = (
  usage: { inputTokens?: number; outputTokens?: number; totalTokens?: number },
): OpenAIUsage => ({
  prompt_tokens: usage.inputTokens ?? 0,
  completion_tokens: usage.outputTokens ?? 0,
  total_tokens: usage.totalTokens ?? (usage.inputTokens ?? 0) + (usage.outputTokens ?? 0),
});
