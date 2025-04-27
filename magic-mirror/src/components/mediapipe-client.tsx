"use client";

import { useEffect, useState } from "react";

import { useHandTracking } from "../hooks/useHandTracking";

const CAROUSEL_ITEMS = [
    { id: 1, color: "red" },
    { id: 2, color: "blue" },
    { id: 3, color: "green" },
    { id: 4, color: "yellow" },
    { id: 5, color: "pink" },
];

export function MediaPipeClient() {
    const { videoRef, canvasRef, isGrabbing, swipeDirection } =
        useHandTracking();
    const [selectedIndex, setSelectedIndex] = useState(
        CAROUSEL_ITEMS.length - 1
    ); // Start with pink selected

    useEffect(() => {
        if (swipeDirection === "right") {
            setSelectedIndex((prev) =>
                prev === 0 ? CAROUSEL_ITEMS.length - 1 : prev - 1
            );
        } else if (swipeDirection === "left") {
            setSelectedIndex((prev) =>
                prev === CAROUSEL_ITEMS.length - 1 ? 0 : prev + 1
            );
        }
    }, [swipeDirection]);

    const getItemPosition = (itemIndex: number) => {
        const diff = itemIndex - selectedIndex;
        const normalizedDiff =
            (diff + CAROUSEL_ITEMS.length) % CAROUSEL_ITEMS.length;
        if (normalizedDiff === 0) return "center";
        if (
            normalizedDiff === 1 ||
            normalizedDiff === -(CAROUSEL_ITEMS.length - 1)
        )
            return "right";
        if (
            normalizedDiff === -1 ||
            normalizedDiff === CAROUSEL_ITEMS.length - 1
        )
            return "left";
        return "hidden";
    };

    return (
        <div className="relative h-full max-h-full w-full max-w-full">
            <video
                ref={videoRef}
                className="hidden"
                playsInline
            />
            <div className="relative flex h-full min-h-full w-full min-w-full items-center justify-center overflow-hidden">
                <canvas
                    ref={canvasRef}
                    className="h-[100vw] scale-x-[-1] rotate-90 object-cover"
                />
            </div>

            {/* <div className="absolute inset-0 flex items-center justify-center">
                <div className="relative w-full max-w-4xl">
                    {CAROUSEL_ITEMS.map((item, index) => {
                        const position = getItemPosition(index);
                        return (
                            <div
                                key={item.id}
                                className={`absolute top-1/2 left-1/2 h-64 w-64 -translate-y-1/2 rounded-xl transition-all duration-500 ${
                                    position === "center"
                                        ? "z-30 -translate-x-1/2 scale-100 opacity-100"
                                        : position === "left"
                                          ? "z-20 -translate-x-[calc(50%+18rem)] scale-75 opacity-50"
                                          : position === "right"
                                            ? "z-20 -translate-x-[calc(50%-18rem)] scale-75 opacity-50"
                                            : "-translate-x-1/2 scale-50 opacity-0"
                                }`}
                                style={{ backgroundColor: item.color }}
                            />
                        );
                    })}
                </div>
            </div> */}

            {isGrabbing && (
                <div className="absolute top-4 left-4 rounded bg-green-500 px-4 py-2 text-white">
                    Grabbing
                </div>
            )}
            {(swipeDirection === "left" || swipeDirection === "right") && (
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-6xl font-bold text-white">
                    Swipe {swipeDirection}
                </div>
            )}
        </div>
    );
}
