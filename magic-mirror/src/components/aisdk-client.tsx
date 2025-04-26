"use client";

import { useState } from "react";
import { useTranscription } from "@/hooks/useTranscription";
import { cn, generateUUID } from "@/lib/utils";
import { useChat } from "@ai-sdk/react";
import { toast } from "sonner";

interface ClientProps {
    readonly id: string;
}

export function AiSDKClient({ id }: ClientProps) {
    const [currentPhrase, setCurrentPhrase] = useState<string>("");

    const {
        messages,
        setMessages,
        handleSubmit,
        input,
        setInput,
        append,
        status,
        stop,
        reload,
        error,
    } = useChat({
        id,
        body: { id },
        experimental_throttle: 100,
        sendExtraMessageFields: true,
        generateId: generateUUID,
        api: "/api/chat",
        onError: (error) => {
            console.error("Error in chat:", error);
            toast.error("An error occurred: " + error?.message);
        },
        onFinish: async (message) => {
            console.log("onFinish", message);
        },
    });

    const { isListening, toggleListening } = useTranscription({
        onTranscript: (transcript, isFinal) => {
            if (isFinal) {
                append({
                    id: generateUUID(),
                    content: transcript,
                    role: "user",
                });
                setCurrentPhrase("");
            } else {
                setCurrentPhrase(transcript);
                stop();
            }
        },
    });

    return (
        <div className="flex min-h-screen flex-col items-center justify-center">
            <h1 className="mb-8 text-3xl font-bold">
                Magic Mirror Transcription
            </h1>

            <button
                onClick={toggleListening}
                className="rounded-full bg-neutral-900 px-6 py-3 text-white transition-colors hover:bg-neutral-700"
            >
                {isListening ? "Stop Listening" : "Start Listening"}
            </button>

            <div className="mt-8 w-full max-w-2xl rounded-lg bg-neutral-100 p-6">
                {/* Current phrase being transcribed */}
                {currentPhrase && (
                    <p className="font-mono text-lg text-blue-600">
                        {currentPhrase}
                    </p>
                )}

                {/* History of transcribed phrases */}
                <div className="mt-4 space-y-2">
                    {messages.map((message) => (
                        <p
                            key={message.id}
                            className={cn(
                                "font-mono text-lg",
                                message.role === "assistant"
                                    ? "text-green-600"
                                    : "text-neutral-900"
                            )}
                        >
                            {message.content}
                        </p>
                    ))}
                </div>
            </div>
        </div>
    );
}
