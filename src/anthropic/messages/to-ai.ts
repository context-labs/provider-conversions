import type {
  ModelMessage,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  AssistantContent,
  UserContent,
} from "ai";
import type { LanguageModelV2ToolResultOutput } from "@ai-sdk/provider";
import type { AiSDKParams } from "~/ai";
import type { MessageCreateParams } from "./types";
import type Anthropic from "@anthropic-ai/sdk";

type AnthropicMessage = Anthropic.MessageParam;
type AnthropicContentBlock = Anthropic.ContentBlockParam;
type AnthropicTool = Anthropic.Tool;
type AnthropicToolChoice = Anthropic.ToolChoice;
type AnthropicTextBlockParam = Anthropic.TextBlockParam;
type AnthropicImageBlockParam = Anthropic.ImageBlockParam;
type AnthropicDocumentBlockParam = Anthropic.DocumentBlockParam;
type AnthropicToolUseBlockParam = Anthropic.ToolUseBlockParam;
type AnthropicToolResultBlockParam = Anthropic.ToolResultBlockParam;
type AnthropicThinkingBlockParam = Anthropic.ThinkingBlockParam;

type UserContentPart =
  | { type: "text"; text: string }
  | { type: "image"; image: string }
  | { type: "file"; data: string; mediaType: string };

export const anthropicMessagesToAiSDK = (
  body: MessageCreateParams,
): AiSDKParams => {
  return {
    model: body.model,
    messages: convertMessages(body.messages),
    system: convertSystem(body.system),
    maxOutputTokens: body.max_tokens,
    temperature: body.temperature ?? undefined,
    topP: body.top_p ?? undefined,
    topK: body.top_k ?? undefined,
    stopSequences: body.stop_sequences ?? undefined,
    tools: convertTools(body.tools),
    toolChoice: convertToolChoice(body.tool_choice),
  };
};

const convertSystem = (
  system: MessageCreateParams["system"],
): string | undefined => {
  if (!system) return undefined;

  if (typeof system === "string") {
    return system;
  }

  // Array of TextBlockParam - concatenate text
  return system.map((block) => block.text).join("\n");
};

const convertMessages = (messages: AnthropicMessage[]): ModelMessage[] => {
  const result: ModelMessage[] = [];

  for (const msg of messages) {
    const converted = convertMessage(msg);
    result.push(...converted);
  }

  return result;
};

const convertMessage = (msg: AnthropicMessage): ModelMessage[] => {
  if (msg.role === "user") {
    return convertUserMessage(msg.content);
  } else {
    return convertAssistantMessage(msg.content);
  }
};

const convertUserMessage = (
  content: AnthropicMessage["content"],
): ModelMessage[] => {
  // String content - simple user message
  if (typeof content === "string") {
    return [{ role: "user", content }];
  }

  // Array content - may contain tool_result blocks mixed with other content
  const userParts: UserContentPart[] = [];
  const toolResults: ToolResultPart[] = [];

  for (const block of content) {
    if (block.type === "tool_result") {
      toolResults.push(convertToolResultBlock(block));
    } else {
      const converted = convertUserContentBlock(block);
      if (converted) {
        userParts.push(converted);
      }
    }
  }

  const messages: ModelMessage[] = [];

  // Tool results come first (respond to previous assistant's tool calls)
  if (toolResults.length > 0) {
    messages.push({
      role: "tool",
      content: toolResults,
    });
  }

  // Then user content
  if (userParts.length > 0) {
    messages.push({
      role: "user",
      content: userParts,
    });
  } else if (toolResults.length === 0) {
    // No tool results and no user parts - add empty user message
    messages.push({
      role: "user",
      content: "",
    });
  }

  return messages;
};

const convertUserContentBlock = (
  block: AnthropicContentBlock,
): UserContentPart | null => {
  switch (block.type) {
    case "text":
      return convertTextBlock(block);

    case "image":
      return convertImageBlock(block);

    case "document":
      return convertDocumentBlock(block);

    case "tool_use":
    case "tool_result":
    case "thinking":
    case "redacted_thinking":
    case "server_tool_use":
    case "web_search_tool_result":
    case "search_result":
      // These are handled separately or not applicable for user content
      return null;

    default: {
      // Exhaustive check for future block types
      const _exhaustive: never = block;
      return null;
    }
  }
};

const convertTextBlock = (block: AnthropicTextBlockParam): UserContentPart => {
  return { type: "text", text: block.text };
};

const convertImageBlock = (
  block: AnthropicImageBlockParam,
): UserContentPart | null => {
  if (block.source.type === "base64") {
    return {
      type: "image",
      image: `data:${block.source.media_type};base64,${block.source.data}`,
    };
  } else if (block.source.type === "url") {
    return {
      type: "image",
      image: block.source.url,
    };
  }
  return null;
};

const convertDocumentBlock = (
  block: AnthropicDocumentBlockParam,
): UserContentPart | null => {
  if (block.source.type === "base64") {
    return {
      type: "file",
      data: block.source.data,
      mediaType: block.source.media_type,
    };
  } else if (block.source.type === "url") {
    // URL documents can't be directly converted to file parts
    return { type: "text", text: `[Document: ${block.source.url}]` };
  } else if (block.source.type === "text") {
    return { type: "text", text: block.source.data };
  } else if (block.source.type === "content") {
    // Content block source - extract text
    const textContent =
      typeof block.source.content === "string"
        ? block.source.content
        : block.source.content
            .filter(
              (c): c is AnthropicTextBlockParam => c.type === "text",
            )
            .map((c) => c.text)
            .join("\n");
    return { type: "text", text: textContent };
  }
  return null;
};

const convertAssistantMessage = (
  content: AnthropicMessage["content"],
): ModelMessage[] => {
  // String content - simple assistant message
  if (typeof content === "string") {
    return [
      {
        role: "assistant",
        content: [{ type: "text", text: content }],
      },
    ];
  }

  // Array content
  const assistantParts: Array<TextPart | ToolCallPart> = [];

  for (const block of content) {
    switch (block.type) {
      case "text":
        assistantParts.push({ type: "text", text: block.text });
        break;

      case "tool_use":
        assistantParts.push(convertToolUseBlock(block));
        break;

      case "thinking":
        assistantParts.push(convertThinkingBlock(block));
        break;

      case "redacted_thinking":
        // Skip redacted thinking - it's encrypted
        break;

      case "image":
      case "document":
      case "tool_result":
      case "server_tool_use":
      case "web_search_tool_result":
      case "search_result":
        // Not typically in assistant messages or handled separately
        break;

      default: {
        const _exhaustive: never = block;
        break;
      }
    }
  }

  const messages: ModelMessage[] = [];

  if (assistantParts.length > 0) {
    messages.push({
      role: "assistant",
      content: assistantParts,
    });
  }

  return messages;
};

const convertToolUseBlock = (block: AnthropicToolUseBlockParam): ToolCallPart => {
  return {
    type: "tool-call",
    toolCallId: block.id,
    toolName: block.name,
    input: block.input,
  };
};

const convertThinkingBlock = (block: AnthropicThinkingBlockParam): TextPart => {
  // AI SDK doesn't have a direct thinking type in AssistantContent
  // Map to text for now
  return { type: "text", text: block.thinking };
};

const convertToolResultBlock = (
  block: AnthropicToolResultBlockParam,
): ToolResultPart => {
  return {
    type: "tool-result",
    toolCallId: block.tool_use_id,
    toolName: "", // Anthropic doesn't include tool name in results
    output: convertToolResultContent(block.content),
  };
};

const convertToolResultContent = (
  content: AnthropicToolResultBlockParam["content"],
): LanguageModelV2ToolResultOutput => {
  if (!content) {
    return { type: "text", value: "" };
  }

  if (typeof content === "string") {
    return { type: "text", value: content };
  }

  // Array of content blocks - extract text
  const textParts = content
    .filter((block): block is AnthropicTextBlockParam => block.type === "text")
    .map((block) => block.text);

  return { type: "text", value: textParts.join("\n") };
};

const convertTools = (
  tools: MessageCreateParams["tools"],
): AiSDKParams["tools"] => {
  if (!tools || tools.length === 0) return undefined;

  const converted: NonNullable<AiSDKParams["tools"]> = {};

  for (const tool of tools) {
    // Only convert custom tools (Tool type), skip built-in tools
    if (isCustomTool(tool)) {
      converted[tool.name] = {
        description: tool.description,
        inputSchema: tool.input_schema,
      };
    }
  }

  return Object.keys(converted).length > 0 ? converted : undefined;
};

const isCustomTool = (
  tool: NonNullable<MessageCreateParams["tools"]>[number],
): tool is AnthropicTool => {
  return "input_schema" in tool;
};

const convertToolChoice = (
  toolChoice: AnthropicToolChoice | undefined,
): AiSDKParams["toolChoice"] => {
  if (!toolChoice) return undefined;

  switch (toolChoice.type) {
    case "auto":
      return "auto";
    case "any":
      return "required";
    case "none":
      return "none";
    case "tool":
      return {
        type: "tool",
        toolName: toolChoice.name,
      };
    default: {
      const _exhaustive: never = toolChoice;
      return undefined;
    }
  }
};
