import type { AiSDKChunk } from "~/ai";
import type { RawMessageStreamEvent, Message } from "./types";
import type { ToolSet } from "ai";
import type Anthropic from "@anthropic-ai/sdk";

type AnthropicContentBlock = Anthropic.ContentBlock;
type AnthropicTextBlock = Anthropic.TextBlock;
type AnthropicToolUseBlock = Anthropic.ToolUseBlock;
type AnthropicStopReason = Anthropic.Message["stop_reason"];
type AnthropicUsage = Anthropic.Usage;
type AnthropicMessageDeltaUsage = Anthropic.MessageDeltaUsage;

export interface ChunkConversionContext {
  /** The response ID to use for all events */
  id: string;
  /** The model name to use */
  model: string;
  /** Track content block indices for streaming */
  contentBlockIndex: number;
  /** Track tool call IDs to their indices */
  toolCallIndices: Map<string, number>;
  /** Track accumulated input JSON per tool call */
  toolCallInputs: Map<string, string>;
  /** Whether message_start has been emitted */
  messageStarted: boolean;
  /** Input tokens (from start event or accumulated) */
  inputTokens: number;
  /** Output tokens accumulated */
  outputTokens: number;
}

export const createChunkConversionContext = (options: {
  id: string;
  model: string;
}): ChunkConversionContext => ({
  id: options.id,
  model: options.model,
  contentBlockIndex: 0,
  toolCallIndices: new Map(),
  toolCallInputs: new Map(),
  messageStarted: false,
  inputTokens: 0,
  outputTokens: 0,
});

export type ChunkConversionResult =
  | { type: "events"; events: RawMessageStreamEvent[] }
  | { type: "skip" }
  | { type: "error"; error: string };

/**
 * Convert an AI SDK stream chunk to Anthropic RawMessageStreamEvent(s).
 *
 * Some AI SDK chunk types don't have direct Anthropic equivalents and will return { type: "skip" }.
 * The context object is mutated to track state across chunks.
 * May return multiple events for a single chunk (e.g., message_start + content_block_start).
 */
export const anthropicChunkFromAiSDK = <T extends ToolSet>(
  chunk: AiSDKChunk<T>,
  context: ChunkConversionContext,
): ChunkConversionResult => {
  const events: RawMessageStreamEvent[] = [];

  switch (chunk.type) {
    case "text-start": {
      // Ensure message_start is emitted first
      if (!context.messageStarted) {
        events.push(createMessageStartEvent(context));
        context.messageStarted = true;
      }

      // Start a new text content block
      events.push(createContentBlockStartEvent(context.contentBlockIndex, {
        type: "text",
        text: "",
        citations: null,
      }));

      return events.length > 0 ? { type: "events", events } : { type: "skip" };
    }

    case "text-delta": {
      // Emit content_block_delta with text_delta
      events.push({
        type: "content_block_delta",
        index: context.contentBlockIndex,
        delta: {
          type: "text_delta",
          text: chunk.text,
        },
      });

      return { type: "events", events };
    }

    case "text-end": {
      // End the current text content block
      events.push({
        type: "content_block_stop",
        index: context.contentBlockIndex,
      });
      context.contentBlockIndex++;

      return { type: "events", events };
    }

    case "reasoning-start": {
      // Ensure message_start is emitted first
      if (!context.messageStarted) {
        events.push(createMessageStartEvent(context));
        context.messageStarted = true;
      }

      // Start a thinking content block
      events.push(createContentBlockStartEvent(context.contentBlockIndex, {
        type: "thinking",
        thinking: "",
        signature: "",
      }));

      return { type: "events", events };
    }

    case "reasoning-delta": {
      // Emit content_block_delta with thinking_delta
      events.push({
        type: "content_block_delta",
        index: context.contentBlockIndex,
        delta: {
          type: "thinking_delta",
          thinking: chunk.text,
        },
      });

      return { type: "events", events };
    }

    case "reasoning-end": {
      // End the current thinking content block
      events.push({
        type: "content_block_stop",
        index: context.contentBlockIndex,
      });
      context.contentBlockIndex++;

      return { type: "events", events };
    }

    case "tool-input-start": {
      // Ensure message_start is emitted first
      if (!context.messageStarted) {
        events.push(createMessageStartEvent(context));
        context.messageStarted = true;
      }

      // Track this tool call
      context.toolCallIndices.set(chunk.id, context.contentBlockIndex);
      context.toolCallInputs.set(chunk.id, "");

      // Start a tool_use content block
      events.push(createContentBlockStartEvent(context.contentBlockIndex, {
        type: "tool_use",
        id: chunk.id,
        name: chunk.toolName,
        input: {},
      }));

      return { type: "events", events };
    }

    case "tool-input-delta": {
      const index = context.toolCallIndices.get(chunk.id);
      if (index === undefined) {
        return { type: "error", error: `Unknown tool call id: ${chunk.id}` };
      }

      // Accumulate input JSON
      const currentInput = context.toolCallInputs.get(chunk.id) ?? "";
      context.toolCallInputs.set(chunk.id, currentInput + chunk.delta);

      // Emit input_json_delta
      events.push({
        type: "content_block_delta",
        index,
        delta: {
          type: "input_json_delta",
          partial_json: chunk.delta,
        },
      });

      return { type: "events", events };
    }

    case "tool-input-end": {
      const index = context.toolCallIndices.get(chunk.id);
      if (index === undefined) {
        return { type: "error", error: `Unknown tool call id: ${chunk.id}` };
      }

      // End the tool_use content block
      events.push({
        type: "content_block_stop",
        index,
      });
      context.contentBlockIndex++;

      return { type: "events", events };
    }

    case "tool-call": {
      // Complete tool call (non-streaming mode)
      // Ensure message_start is emitted first
      if (!context.messageStarted) {
        events.push(createMessageStartEvent(context));
        context.messageStarted = true;
      }

      const index = context.contentBlockIndex;
      context.toolCallIndices.set(chunk.toolCallId, index);

      // Start the tool_use block
      events.push(createContentBlockStartEvent(index, {
        type: "tool_use",
        id: chunk.toolCallId,
        name: chunk.toolName,
        input: {},
      }));

      // Send the full input as a delta
      const inputJson = JSON.stringify(chunk.input);
      events.push({
        type: "content_block_delta",
        index,
        delta: {
          type: "input_json_delta",
          partial_json: inputJson,
        },
      });

      // End the block
      events.push({
        type: "content_block_stop",
        index,
      });
      context.contentBlockIndex++;

      return { type: "events", events };
    }

    case "tool-result":
    case "tool-error":
      // Tool results are not part of the assistant's stream response
      return { type: "skip" };

    case "finish-step": {
      // Update usage from step finish
      if (chunk.usage.inputTokens !== undefined) {
        context.inputTokens = chunk.usage.inputTokens;
      }
      if (chunk.usage.outputTokens !== undefined) {
        context.outputTokens = chunk.usage.outputTokens;
      }

      // Skip - we'll emit message_delta on "finish"
      return { type: "skip" };
    }

    case "finish": {
      // Ensure message_start was emitted
      if (!context.messageStarted) {
        events.push(createMessageStartEvent(context));
        context.messageStarted = true;
      }

      // Emit message_delta with stop_reason and usage
      events.push({
        type: "message_delta",
        delta: {
          stop_reason: convertStopReason(chunk.finishReason),
          stop_sequence: null,
        },
        usage: {
          output_tokens: chunk.totalUsage.outputTokens ?? context.outputTokens,
          input_tokens: chunk.totalUsage.inputTokens ?? context.inputTokens,
          cache_creation_input_tokens: null,
          cache_read_input_tokens: null,
          server_tool_use: null,
        },
      });

      // Emit message_stop
      events.push({
        type: "message_stop",
      });

      return { type: "events", events };
    }

    case "start": {
      // Emit message_start if we have request info
      if (!context.messageStarted) {
        events.push(createMessageStartEvent(context));
        context.messageStarted = true;
      }
      return events.length > 0 ? { type: "events", events } : { type: "skip" };
    }

    case "start-step":
      // No direct equivalent in Anthropic streaming
      return { type: "skip" };

    case "abort":
      // No direct equivalent
      return { type: "skip" };

    case "error":
      return { type: "error", error: String(chunk.error) };

    case "source":
    case "file":
    case "raw":
      // No direct equivalents in Anthropic
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

const createMessageStartEvent = (
  context: ChunkConversionContext,
): Anthropic.RawMessageStartEvent => ({
  type: "message_start",
  message: {
    id: context.id,
    type: "message",
    role: "assistant",
    model: context.model as Anthropic.Model,
    content: [],
    stop_reason: null,
    stop_sequence: null,
    usage: {
      input_tokens: context.inputTokens,
      output_tokens: 0,
      cache_creation_input_tokens: null,
      cache_read_input_tokens: null,
      cache_creation: null,
      server_tool_use: null,
      service_tier: null,
    },
  },
});

const createContentBlockStartEvent = (
  index: number,
  contentBlock: AnthropicContentBlock,
): Anthropic.RawContentBlockStartEvent => ({
  type: "content_block_start",
  index,
  content_block: contentBlock,
});

const convertStopReason = (
  finishReason: string,
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
    default:
      return "end_turn";
  }
};
