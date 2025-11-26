# @inference-net/provider-conversions

Convert between LLM provider API formats (OpenAI, Anthropic, Google GenAI) and the Vercel AI SDK format.

## Installation

```bash
npm install @inference-net/provider-conversions
```

### Peer Dependencies

You'll need to install the provider SDKs you plan to use:

```bash
# For OpenAI conversions
npm install openai

# For Anthropic conversions
npm install @anthropic-ai/sdk

# For Google GenAI conversions
npm install @google/genai

# For AI SDK types (required)
npm install ai @ai-sdk/provider
```

## Usage

### OpenAI Chat Completions

```typescript
import { OpenAIChatCompletion } from "@inference-net/provider-conversions";

// Convert OpenAI request → AI SDK format
const aiRequest = OpenAIChatCompletion.openaiToAiSDK(openaiRequest);

// Convert AI SDK response → OpenAI format
const openaiResponse = OpenAIChatCompletion.openaiFromAiSDK(aiResponse, {
  requestId: "req_123",
  model: "gpt-4",
  created: Date.now(),
});

// Convert AI SDK stream chunks → OpenAI format
const context = OpenAIChatCompletion.createChunkConversionContext({
  requestId: "req_123",
  model: "gpt-4",
  created: Date.now(),
});

for await (const chunk of aiStream) {
  const result = OpenAIChatCompletion.openaiChunkFromAiSDK(chunk, context);
  if (result.type === "chunk") {
    // Send result.chunk to client
  }
}
```

### Anthropic Messages

```typescript
import { AnthropicMessages } from "@inference-net/provider-conversions";

// Convert Anthropic request → AI SDK format
const aiRequest = AnthropicMessages.anthropicToAiSDK(anthropicRequest);

// Convert AI SDK response → Anthropic format
const anthropicResponse = AnthropicMessages.anthropicFromAiSDK(aiResponse, {
  responseId: "msg_123",
  model: "claude-3-opus-20240229",
});

// Convert AI SDK stream chunks → Anthropic format
const context = AnthropicMessages.createChunkConversionContext({
  responseId: "msg_123",
  model: "claude-3-opus-20240229",
});

for await (const chunk of aiStream) {
  const result = AnthropicMessages.anthropicChunkFromAiSDK(chunk, context);
  if (result.type === "events") {
    for (const event of result.events) {
      // Send SSE event to client
    }
  }
}
```

### Google GenAI

```typescript
import { GoogleGenAI } from "@inference-net/provider-conversions";

// Convert Google GenAI request → AI SDK format
const aiRequest = GoogleGenAI.googleGenaiToAiSDK(googleRequest);

// Convert AI SDK response → Google GenAI format
const googleResponse = GoogleGenAI.googleGenaiFromAiSDK(aiResponse, {
  responseId: "resp_123",
  modelVersion: "gemini-1.5-pro",
});

// Convert AI SDK stream chunks → Google GenAI format
const context = GoogleGenAI.createChunkConversionContext({
  responseId: "resp_123",
  modelVersion: "gemini-1.5-pro",
});

for await (const chunk of aiStream) {
  const result = GoogleGenAI.googleChunkFromAiSDK(chunk, context);
  if (result.type === "response") {
    // Send result.response to client
  }
}
```

## Tool/Function Calling

All providers support tool/function calling conversions:

```typescript
// OpenAI tools → AI SDK tools
const openaiRequest = {
  model: "gpt-4",
  messages: [{ role: "user", content: "What's the weather?" }],
  tools: [{
    type: "function",
    function: {
      name: "get_weather",
      description: "Get current weather",
      parameters: {
        type: "object",
        properties: {
          location: { type: "string" }
        },
        required: ["location"]
      }
    }
  }]
};

const aiRequest = OpenAIChatCompletion.openaiToAiSDK(openaiRequest);
// aiRequest.tools contains the converted tool definitions
```

## Type Exports

Access AI SDK types for building your own conversions:

```typescript
import { AISDKTypes } from "@inference-net/provider-conversions";

type Message = AISDKTypes.AiSDKMessage;
type Response = AISDKTypes.AiSDKResponse;
type Chunk = AISDKTypes.AiSDKChunk;
```

## API Reference

| Provider | Request Conversion | Response Conversion | Stream Conversion |
|----------|-------------------|--------------------|--------------------|
| OpenAI | `openaiToAiSDK()` | `openaiFromAiSDK()` | `openaiChunkFromAiSDK()` |
| Anthropic | `anthropicToAiSDK()` | `anthropicFromAiSDK()` | `anthropicChunkFromAiSDK()` |
| Google GenAI | `googleGenaiToAiSDK()` | `googleGenaiFromAiSDK()` | `googleChunkFromAiSDK()` |

## License

MIT
