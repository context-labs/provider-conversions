import type {
  ModelMessage,
  TextPart,
  ToolCallPart,
  ToolResultPart,
} from "ai";
import type { LanguageModelV2ToolResultOutput } from "@ai-sdk/provider";
import type { AiSDKParams } from "~/ai";
import type { GenerateContentParameters } from "./types";
import type * as GenAI from "@google/genai";

type GoogleContent = GenAI.Content;
type GooglePart = GenAI.Part;
type GoogleTool = GenAI.Tool;
type GoogleFunctionDeclaration = GenAI.FunctionDeclaration;
type GoogleFunctionCallingConfig = GenAI.FunctionCallingConfig;

type UserContentPart =
  | { type: "text"; text: string }
  | { type: "image"; image: string }
  | { type: "file"; data: string; mediaType: string };

export const googleGenaiToAiSDK = (
  params: GenerateContentParameters,
): AiSDKParams => {
  return {
    model: params.model,
    messages: convertContents(params.contents),
    system: convertSystemInstruction(params.config?.systemInstruction),
    maxOutputTokens: params.config?.maxOutputTokens ?? undefined,
    temperature: params.config?.temperature ?? undefined,
    topP: params.config?.topP ?? undefined,
    topK: params.config?.topK ?? undefined,
    stopSequences: params.config?.stopSequences ?? undefined,
    presencePenalty: params.config?.presencePenalty ?? undefined,
    frequencyPenalty: params.config?.frequencyPenalty ?? undefined,
    seed: params.config?.seed ?? undefined,
    tools: convertTools(params.config?.tools),
    toolChoice: convertToolChoice(params.config?.toolConfig?.functionCallingConfig),
  };
};

const convertSystemInstruction = (
  systemInstruction: GenAI.ContentUnion | undefined,
): string | undefined => {
  if (!systemInstruction) return undefined;

  // ContentUnion can be Content | PartUnion[] | PartUnion
  if (typeof systemInstruction === "string") {
    return systemInstruction;
  }

  // If it's a Content object
  if ("parts" in systemInstruction && systemInstruction.parts) {
    return systemInstruction.parts
      .filter((part): part is GenAI.Part & { text: string } => "text" in part && typeof part.text === "string")
      .map((part) => part.text)
      .join("\n");
  }

  // If it's a Part or array of Parts
  if (Array.isArray(systemInstruction)) {
    return systemInstruction
      .filter((part): part is GenAI.Part & { text: string } =>
        typeof part === "object" && part !== null && "text" in part && typeof part.text === "string"
      )
      .map((part) => part.text)
      .join("\n");
  }

  // Single Part with text
  if (
    typeof systemInstruction === "object" &&
    "text" in systemInstruction &&
    typeof systemInstruction.text === "string"
  ) {
    return systemInstruction.text;
  }

  return undefined;
};

const convertContents = (
  contents: GenAI.ContentListUnion,
): ModelMessage[] => {
  // ContentListUnion = Content | Content[] | PartUnion | PartUnion[]
  const contentArray = normalizeToContentArray(contents);
  const result: ModelMessage[] = [];

  for (const content of contentArray) {
    const converted = convertContent(content);
    result.push(...converted);
  }

  return result;
};

const isContent = (value: GenAI.ContentListUnion): value is GoogleContent => {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value) &&
    "role" in value
  );
};

const isContentArray = (
  value: GenAI.ContentListUnion,
): value is GoogleContent[] => {
  if (!Array.isArray(value) || value.length === 0) {
    return false;
  }
  const first = value[0];
  return typeof first === "object" && first !== null && "role" in first;
};

const partUnionToPart = (partUnion: GenAI.PartUnion): GooglePart => {
  if (typeof partUnion === "string") {
    return { text: partUnion };
  }
  return partUnion;
};

const normalizeToContentArray = (
  contents: GenAI.ContentListUnion,
): GoogleContent[] => {
  // If it's a single Content object with role
  if (isContent(contents)) {
    return [contents];
  }

  // If it's an array of Content objects
  if (isContentArray(contents)) {
    return contents;
  }

  // If it's an array of PartUnion - wrap in user content
  if (Array.isArray(contents)) {
    const parts: GooglePart[] = contents.map(partUnionToPart);
    return [{ role: "user", parts }];
  }

  // Single PartUnion - wrap in user content
  // At this point contents is PartUnion (Part | string)
  const part = partUnionToPart(contents);
  return [{ role: "user", parts: [part] }];
};

const convertContent = (content: GoogleContent): ModelMessage[] => {
  const role = content.role;

  if (role === "user") {
    return convertUserContent(content);
  } else if (role === "model") {
    return convertModelContent(content);
  }

  // Unknown role - treat as user
  return convertUserContent(content);
};

const convertUserContent = (content: GoogleContent): ModelMessage[] => {
  const messages: ModelMessage[] = [];
  const userParts: UserContentPart[] = [];
  const toolResults: ToolResultPart[] = [];

  for (const part of content.parts ?? []) {
    if (part.functionResponse) {
      toolResults.push(convertFunctionResponsePart(part));
    } else {
      const converted = convertUserPart(part);
      if (converted) {
        userParts.push(converted);
      }
    }
  }

  // Tool results come first
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

const convertUserPart = (part: GooglePart): UserContentPart | null => {
  // Text part
  if (part.text !== undefined) {
    return { type: "text", text: part.text };
  }

  // Inline data (image or file)
  if (part.inlineData) {
    const mimeType = part.inlineData.mimeType ?? "application/octet-stream";
    const data = part.inlineData.data ?? "";

    if (mimeType.startsWith("image/")) {
      return {
        type: "image",
        image: `data:${mimeType};base64,${data}`,
      };
    }

    return {
      type: "file",
      data,
      mediaType: mimeType,
    };
  }

  // File data (URI reference)
  if (part.fileData) {
    // Can't directly convert URI to file content, represent as text
    return {
      type: "text",
      text: `[File: ${part.fileData.fileUri}]`,
    };
  }

  // Skip function calls in user content (shouldn't happen)
  // Skip executable code, code execution results (not user content)
  return null;
};

const convertFunctionResponsePart = (part: GooglePart): ToolResultPart => {
  const functionResponse = part.functionResponse!;
  return {
    type: "tool-result",
    toolCallId: functionResponse.id ?? functionResponse.name ?? "",
    toolName: functionResponse.name ?? "",
    output: convertFunctionResponseOutput(functionResponse.response),
  };
};

const convertFunctionResponseOutput = (
  response: Record<string, unknown> | undefined,
): LanguageModelV2ToolResultOutput => {
  if (!response) {
    return { type: "text", value: "" };
  }

  // If response has an "output" key, use that
  if ("output" in response) {
    return { type: "text", value: String(response.output) };
  }

  // If response has an "error" key, use that
  if ("error" in response) {
    return { type: "text", value: String(response.error) };
  }

  // Otherwise stringify the whole response
  return { type: "text", value: JSON.stringify(response) };
};

const convertModelContent = (content: GoogleContent): ModelMessage[] => {
  const assistantParts: Array<TextPart | ToolCallPart> = [];

  for (const part of content.parts ?? []) {
    // Text part
    if (part.text !== undefined && !part.thought) {
      assistantParts.push({ type: "text", text: part.text });
    }

    // Thought/thinking text (map to regular text for now)
    if (part.text !== undefined && part.thought) {
      assistantParts.push({ type: "text", text: part.text });
    }

    // Function call
    if (part.functionCall) {
      assistantParts.push(convertFunctionCallPart(part));
    }
  }

  if (assistantParts.length === 0) {
    return [];
  }

  return [
    {
      role: "assistant",
      content: assistantParts,
    },
  ];
};

const convertFunctionCallPart = (part: GooglePart): ToolCallPart => {
  const functionCall = part.functionCall!;
  return {
    type: "tool-call",
    toolCallId: functionCall.id ?? functionCall.name ?? "",
    toolName: functionCall.name ?? "",
    input: functionCall.args ?? {},
  };
};

const convertTools = (
  tools: GenAI.ToolListUnion | undefined,
): AiSDKParams["tools"] => {
  if (!tools || tools.length === 0) return undefined;

  const converted: NonNullable<AiSDKParams["tools"]> = {};

  for (const tool of tools) {
    // Tool can be Tool or CallableTool
    // We only handle Tool with functionDeclarations
    if ("functionDeclarations" in tool && tool.functionDeclarations) {
      for (const funcDecl of tool.functionDeclarations) {
        if (funcDecl.name) {
          converted[funcDecl.name] = {
            description: funcDecl.description,
            inputSchema: convertFunctionParameters(funcDecl),
          };
        }
      }
    }
  }

  return Object.keys(converted).length > 0 ? converted : undefined;
};

const isJsonSchemaObject = (
  value: unknown,
): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null && !Array.isArray(value);
};

const convertFunctionParameters = (
  funcDecl: GoogleFunctionDeclaration,
): Record<string, unknown> => {
  // Prefer parametersJsonSchema if available (typed as unknown in SDK)
  if (
    funcDecl.parametersJsonSchema !== undefined &&
    isJsonSchemaObject(funcDecl.parametersJsonSchema)
  ) {
    return funcDecl.parametersJsonSchema;
  }

  // Convert Schema to JSON Schema format
  if (funcDecl.parameters) {
    return convertSchemaToJsonSchema(funcDecl.parameters);
  }

  // No parameters
  return { type: "object", properties: {} };
};

const convertSchemaToJsonSchema = (
  schema: GenAI.Schema,
): Record<string, unknown> => {
  const result: Record<string, unknown> = {};

  if (schema.type) {
    // Google uses uppercase types like "STRING", "OBJECT", etc.
    result.type = schema.type.toLowerCase();
  }

  if (schema.description) {
    result.description = schema.description;
  }

  if (schema.enum) {
    result.enum = schema.enum;
  }

  if (schema.items) {
    result.items = convertSchemaToJsonSchema(schema.items);
  }

  if (schema.properties) {
    const props: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(schema.properties)) {
      props[key] = convertSchemaToJsonSchema(value);
    }
    result.properties = props;
  }

  if (schema.required) {
    result.required = schema.required;
  }

  return result;
};

const convertToolChoice = (
  functionCallingConfig: GoogleFunctionCallingConfig | undefined,
): AiSDKParams["toolChoice"] => {
  if (!functionCallingConfig) return undefined;

  const mode = functionCallingConfig.mode;

  switch (mode) {
    case "AUTO":
    case "MODE_UNSPECIFIED":
    case "VALIDATED":
      return "auto";

    case "ANY":
      // If specific function names are allowed, use tool choice
      if (
        functionCallingConfig.allowedFunctionNames &&
        functionCallingConfig.allowedFunctionNames.length === 1
      ) {
        return {
          type: "tool",
          toolName: functionCallingConfig.allowedFunctionNames[0],
        };
      }
      return "required";

    case "NONE":
      return "none";

    default:
      return undefined;
  }
};
