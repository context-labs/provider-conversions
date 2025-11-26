import { googleGenaiFromAiSDK } from "./from-ai";
import type { FromAiSDKOptions } from "./from-ai";
import { googleGenaiToAiSDK } from "./to-ai";
import {
  googleChunkFromAiSDK,
  createChunkConversionContext,
} from "./chunk-from-ai";
import type {
  ChunkConversionContext,
  ChunkConversionResult,
} from "./chunk-from-ai";
import type {
  GenerateContentParameters,
  GenerateContentResponse,
  GenerateContentConfig,
} from "./types";

export {
  // Request conversion (Google GenAI -> AI SDK)
  googleGenaiToAiSDK,
  // Response conversion (AI SDK -> Google GenAI)
  googleGenaiFromAiSDK,
  // Chunk/streaming conversion (AI SDK -> Google GenAI)
  googleChunkFromAiSDK,
  createChunkConversionContext,
  // Types
  type FromAiSDKOptions,
  type ChunkConversionContext,
  type ChunkConversionResult,
  type GenerateContentParameters,
  type GenerateContentResponse,
  type GenerateContentConfig,
};
