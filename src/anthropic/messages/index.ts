import { anthropicMessagesFromAiSDK } from "./from-ai";
import type { FromAiSDKOptions } from "./from-ai";
import { anthropicMessagesToAiSDK } from "./to-ai";
import {
  anthropicChunkFromAiSDK,
  createChunkConversionContext,
} from "./chunk-from-ai";
import type {
  ChunkConversionContext,
  ChunkConversionResult,
} from "./chunk-from-ai";
import type {
  MessageCreateParams,
  Message,
  RawMessageStreamEvent,
} from "./types";

export {
  // Request conversion (Anthropic -> AI SDK)
  anthropicMessagesToAiSDK,
  // Response conversion (AI SDK -> Anthropic)
  anthropicMessagesFromAiSDK,
  // Chunk/streaming conversion (AI SDK -> Anthropic)
  anthropicChunkFromAiSDK,
  createChunkConversionContext,
  // Types
  type FromAiSDKOptions,
  type ChunkConversionContext,
  type ChunkConversionResult,
  type MessageCreateParams,
  type Message,
  type RawMessageStreamEvent,
};
