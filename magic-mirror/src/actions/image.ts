"use server";

import { OpenAI } from "openai";

export async function generateImage(
    imageFile: File | null,
    prompt: string,
    quality: "low" | "medium" = "medium"
): Promise<{ success: boolean; imageBase64: string }> {
    const openai = new OpenAI({ apiKey: process.env.IMAGE_OPENAI_API_KEY });

    const startTime = Date.now();

    const result = imageFile
        ? await openai.images.edit({
              model: "gpt-image-1",
              image: [imageFile],
              prompt: prompt,
              size: "1024x1024",
              quality: quality,
          })
        : await openai.images.generate({
              model: "gpt-image-1",
              prompt: prompt,
              size: "1024x1536",
              quality: quality,
          });

    const endTime = Date.now();
    console.log(`Image API returned in ${endTime - startTime}ms`);

    if (!result.data?.[0]?.b64_json) {
        throw new Error("Failed to get image data from OpenAI");
    }

    const imageBase64 = result.data[0].b64_json;
    return {
        success: true,
        imageBase64: imageBase64,
    };
}
