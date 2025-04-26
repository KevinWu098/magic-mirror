import { SYSTEM_PROMPT } from "@/lib/system";
import { generateUUID, getMostRecentUserMessage } from "@/lib/utils";
import { openai } from "@ai-sdk/openai";
import {
    appendResponseMessages,
    createDataStreamResponse,
    smoothStream,
    streamText,
    UIMessage,
} from "ai";

export async function POST(request: Request) {
    try {
        const {
            id,
            messages,
        }: {
            id: string;
            messages: Array<UIMessage>;
        } = await request.json();

        const userMessage = getMostRecentUserMessage(messages);

        if (!userMessage) {
            return new Response("No user message found", { status: 400 });
        }

        return createDataStreamResponse({
            execute: (dataStream) => {
                const result = streamText({
                    model: openai("gpt-4o-mini"),
                    system: SYSTEM_PROMPT,
                    messages,
                    maxSteps: 5,
                    experimental_activeTools: [
                        "generateExtension",
                        "getPageContext",
                    ],
                    experimental_transform: smoothStream({ chunking: "word" }),
                    experimental_generateMessageId: generateUUID,
                    toolCallStreaming: true,
                    // tools: {
                    //     generateExtension: generateExtension(),
                    //     getPageContext,
                    // },
                    onFinish: async ({ response }) => {
                        try {
                            const assistantId = response.messages
                                .filter(
                                    (message) => message.role === "assistant"
                                )
                                .at(-1)?.id;

                            if (!assistantId) {
                                throw new Error("No assistant message found!");
                            }

                            const [, assistantMessage] = appendResponseMessages(
                                {
                                    messages: [userMessage],
                                    responseMessages: response.messages,
                                }
                            );

                            if (!assistantMessage) {
                                throw new Error("No assistant message found!");
                            }
                        } catch (_) {
                            console.error("Failed to save chat");
                        }
                    },
                });

                result.consumeStream();

                result.mergeIntoDataStream(dataStream, {
                    sendReasoning: true,
                });
            },
            onError: () => {
                return "Oops, an error occurred!";
            },
        });
    } catch {
        return new Response(
            "An error occurred while processing your request!",
            {
                status: 404,
            }
        );
    }
}
