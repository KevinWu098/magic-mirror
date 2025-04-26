"use client";

import { useHandTracking } from "../hooks/useHandTracking";

export function Client() {
    const { videoRef, canvasRef, isGrabbing, swipeDirection } =
        useHandTracking();

    return (
        <div className="relative h-full">
            <video
                ref={videoRef}
                className="hidden"
                playsInline
            />
            <canvas
                ref={canvasRef}
                className="h-full w-full scale-x-[-1] object-cover"
            />
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
