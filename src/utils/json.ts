/**
 * Safely parse a JSON string, returning the original string if parsing fails.
 */
export const safeJsonParse = (json: string): unknown => {
  try {
    return JSON.parse(json);
  } catch {
    return json;
  }
};
