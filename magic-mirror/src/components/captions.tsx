import React from "react";
import { cn } from "@/lib/utils";

interface Message {
    role: "assistant" | "user";
    text: string;
}

interface CaptionsProps {
    messages?: Message[];
    userIsFinal?: boolean;
}

export const Captions = ({
    messages = [],
    userIsFinal = false,
}: CaptionsProps) => {
    const latest = messages.at(-1);
    if (!latest) return null;

    return (
        <div className="pointer-events-none fixed top-50 left-0 z-[100] flex w-full flex-col items-center">
            <div className="mb-4 flex w-[90vw] flex-col gap-4 rounded-xl bg-black/30 px-8 py-6 shadow-lg backdrop-blur-sm">
                {latest.role === "assistant" && (
                    <div className="line-clamp-3 text-center text-6xl font-semibold text-ellipsis text-blue-200">
                        {latest.text}
                    </div>
                )}
                {latest.role === "user" && (
                    <div
                        className={cn(
                            "line-clamp-3 text-center text-6xl font-medium text-ellipsis",
                            userIsFinal
                                ? "text-white"
                                : "animate-pulse text-green-300 italic"
                        )}
                    >
                        {latest.text}
                    </div>
                )}
            </div>
        </div>
    );
};
