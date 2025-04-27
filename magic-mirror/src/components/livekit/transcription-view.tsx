import * as React from "react";
import { useCombinedTranscriptions } from "@/hooks/useCombinedTranscriptions";

export function TranscriptionView() {
    const combinedTranscriptions = useCombinedTranscriptions();
    const containerRef = React.useRef<HTMLDivElement>(null);

    // scroll to bottom when new transcription is added
    React.useEffect(() => {
        if (containerRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
    }, [combinedTranscriptions]);

    return (
        <div className="relative mx-auto h-[200px] w-[512px] max-w-[90vw]">
            {/* Fade-out gradient mask */}
            <div className="pointer-events-none absolute top-0 right-0 left-0 z-10 h-8 bg-gradient-to-b from-[var(--lk-bg)] to-transparent" />
            <div className="pointer-events-none absolute right-0 bottom-0 left-0 z-10 h-8 bg-gradient-to-t from-[var(--lk-bg)] to-transparent" />

            {/* Scrollable content */}
            <div
                ref={containerRef}
                className="flex h-full flex-col gap-2 overflow-y-auto px-4 py-8"
            >
                {combinedTranscriptions.map((segment) => (
                    <div
                        id={segment.id}
                        key={segment.id}
                        className={
                            segment.role === "assistant"
                                ? "fit-content self-start p-2"
                                : "fit-content self-end rounded-md bg-gray-800 p-2"
                        }
                    >
                        {segment.text}
                    </div>
                ))}
            </div>
        </div>
    );
}
