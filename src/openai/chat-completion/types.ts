import type OpenAI from "openai";

export type ChatCompletionRequestBody = OpenAI.ChatCompletionCreateParams;
export type ChatCompletionResponse = OpenAI.ChatCompletion;
export type ChatCompletionChunk = OpenAI.ChatCompletionChunk;
