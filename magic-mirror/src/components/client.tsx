"use client";

import { useEffect, useState } from "react";
import { takeCalibrationImage } from "@/actions";
import { ConnectionDetails } from "@/app/api/connection-details/route";
import { Captions } from "@/components/captions";
import { TranscriptionView } from "@/components/livekit/transcription-view";
import { useHandTracking } from "@/hooks/useHandTracking";
import { onDeviceFailure } from "@/lib/livekit";
import {
    ControlBar,
    RoomAudioRenderer,
    RoomContext,
    useVoiceAssistant,
} from "@livekit/components-react";
import { Room, RoomEvent, RpcError, RpcInvocationData } from "livekit-client";

// Message type
interface Message {
    role: "assistant" | "user";
    text: string;
}

export function Client() {
    const { videoRef, canvasRef, isGrabbing, swipeDirection } =
        useHandTracking();

    const [room] = useState(new Room());

    // New state for captions
    const [messages, setMessages] = useState<Message[]>([]);
    const [userIsFinal, setUserIsFinal] = useState<boolean>(true);

    useEffect(() => {
        async function connect() {
            const url = new URL(
                process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT ??
                    "/api/connection-details",
                window.location.origin
            );
            const response = await fetch(url.toString());
            const connectionDetailsData: ConnectionDetails =
                await response.json();

            await room.connect(
                connectionDetailsData.serverUrl,
                connectionDetailsData.participantToken
            );
            await room.localParticipant.setMicrophoneEnabled(true);
        }

        connect();

        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        room.on(RoomEvent.MediaDevicesError, onDeviceFailure);

        return () => {
            room.off(RoomEvent.MediaDevicesError, onDeviceFailure);
        };
    }, [room]);

    const localParticipant = room.localParticipant;

    useEffect(() => {
        localParticipant.registerRpcMethod(
            "takeCalibrationImage",
            async (data: RpcInvocationData) => {
                try {
                    if (!videoRef.current) {
                        throw new RpcError(1, "Video element not found");
                    }

                    const canvas = document.createElement("canvas");
                    canvas.width = videoRef.current.videoWidth;
                    canvas.height = videoRef.current.videoHeight;

                    const ctx = canvas.getContext("2d");
                    if (!ctx) {
                        throw new RpcError(
                            1,
                            "Could not create canvas context"
                        );
                    }
                    const response = canvas.toDataURL("image/jpeg", 0.5);

                    const { success } = await takeCalibrationImage(response);

                    return success.toString();
                } catch (error) {
                    throw new RpcError(1, "Failed to capture video frame");
                }
            }
        );

        room.on(
            RoomEvent.TranscriptionReceived,
            (segments, participantInfo) => {
                for (const segment of segments) {
                    const isAssistant =
                        !participantInfo ||
                        participantInfo?.identity.includes("agent");

                    setMessages((prev) => [
                        ...prev,
                        {
                            role: isAssistant ? "assistant" : "user",
                            text: segment.text,
                        },
                    ]);
                    if (!isAssistant) {
                        setUserIsFinal(false);
                    }
                }
            }
        );

        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return (
        <div className="lk-room-container relative mx-auto h-full max-h-full w-full max-w-full overflow-hidden">
            <RoomContext.Provider value={room}>
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

                <div className="absolute bottom-8 z-50 flex w-full flex-row items-center justify-center">
                    <ControlBar />
                </div>
                {/* <TranscriptionView /> */}
                <Captions
                    messages={messages}
                    userIsFinal={userIsFinal}
                />
                <RoomAudioRenderer />
            </RoomContext.Provider>
        </div>
    );
}
