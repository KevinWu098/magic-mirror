"server-only";

import { z } from "zod";

interface GoogleLensResponse {
  [key: string]: any;
}

const searchParamsSchema = z.object({
  imageUrl: z.string().url(),
  country: z.string().optional().default("us"),
});

export async function searchGoogleLens(
  params: z.infer<typeof searchParamsSchema>
): Promise<GoogleLensResponse | { error: string }> {
  try {
    const { imageUrl, country } = searchParamsSchema.parse(params);
    const apiKey = process.env.ZYLA_GOOGLE_LENS_API_KEY;
    
    if (!apiKey) {
      throw new Error("API key not found. Make sure ZYLA_GOOGLE_LENS_API_KEY is set in environment variables");
    }
    
    const endpoint = "https://zylalabs.com/api/1338/google+lens+search+api/1119/search";
    
    const url = new URL(endpoint);
    url.searchParams.append("url", imageUrl);
    url.searchParams.append("country", country);
    
    const requestHeaders = {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    };
    
    const response = await fetch(url.toString(), {
      method: "GET",
      headers: requestHeaders,
      cache: "no-store",
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Error: ${response.status}`);
      console.error(errorText);
      return { error: `API request failed with status ${response.status}` };
    }
    
    const result = await response.json();
    return result;
    
  } catch (error) {
    console.error("Error searching Google Lens:", error);
    return { 
      error: error instanceof Error ? error.message : "An unknown error occurred" 
    };
  }
}

