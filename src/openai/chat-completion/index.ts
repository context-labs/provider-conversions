import { openaiChatCompletionFromAiSDK } from "./from-ai";
import type { FromAiSDKOptions } from "./from-ai";
import { openaiChatCompletionToAiSDK } from "./to-ai";
import {
  openaiChunkFromAiSDK,
  createChunkConversionContext,
} from "./chunk-from-ai";
import type {
  ChunkConversionContext,
  ChunkConversionResult,
} from "./chunk-from-ai";
import type {
  ChatCompletionRequestBody,
  ChatCompletionResponse,
  ChatCompletionChunk,
} from "./types";

export {
  // Request conversion (OpenAI -> AI SDK)
  openaiChatCompletionToAiSDK,
  // Response conversion (AI SDK -> OpenAI)
  openaiChatCompletionFromAiSDK,
  // Chunk/streaming conversion (AI SDK -> OpenAI)
  openaiChunkFromAiSDK,
  createChunkConversionContext,
  // Types
  type FromAiSDKOptions,
  type ChunkConversionContext,
  type ChunkConversionResult,
  type ChatCompletionRequestBody,
  type ChatCompletionResponse,
  type ChatCompletionChunk,
};
