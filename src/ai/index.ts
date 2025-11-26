import {
  generateText,
  streamText,
  type ToolSet,
  type TextStreamPart,
} from "ai";

type RelevantAISDKParamsFields =
  // Core
  | "model"
  | "messages"
  // Prompt
  | "system"
  | "prompt"
  // Generation params
  | "maxOutputTokens"
  | "temperature"
  | "topP"
  | "topK"
  | "frequencyPenalty"
  | "presencePenalty"
  | "stopSequences"
  | "seed"
  // Tools
  | "tools"
  | "toolChoice";

type RelevantAISDKResponseFields =
  // Content
  | "text"
  | "reasoning"
  | "reasoningText"
  | "files"
  | "sources"
  // Tool calls & results
  | "toolCalls"
  | "toolResults"
  // Completion info
  | "finishReason"
  | "usage"
  // Response metadata
  | "response";

export type AiSDKGenerateTextParams = Parameters<typeof generateText>[0];
export type AiSDKGenerateTextReturn = Awaited<ReturnType<typeof generateText>>;

export type AiSDKStreamTextParams = Parameters<typeof streamText>[0];
export type AiSDKStreamTextReturn = Awaited<ReturnType<typeof streamText>>;

export type AiSDKParams = Pick<
  AiSDKGenerateTextParams & AiSDKStreamTextParams,
  RelevantAISDKParamsFields
>;

export type AiSDKResponse = Pick<
  AiSDKGenerateTextReturn,
  RelevantAISDKResponseFields
>;

export type AiSDKChunk<T extends ToolSet> = TextStreamPart<T>;
