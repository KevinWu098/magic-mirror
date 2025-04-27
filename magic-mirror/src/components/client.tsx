"use client";

import { useEffect, useState } from "react";
import { takeCalibrationImage } from "@/actions";
import { ConnectionDetails } from "@/app/api/connection-details/route";
import { Captions } from "@/components/captions";
import { TranscriptionView } from "@/components/livekit/transcription-view";
import { Button } from "@/components/ui/button";
import { useHandTracking } from "@/hooks/useHandTracking";
import { onDeviceFailure } from "@/lib/livekit";
import {
    RoomAudioRenderer,
    RoomContext,
    useVoiceAssistant,
} from "@livekit/components-react";
import {
    Room,
    RoomEvent,
    RpcError,
    RpcInvocationData,
    Track,
} from "livekit-client";
import { Mic2Icon, MicOffIcon } from "lucide-react";

// Message type
interface Message {
    role: "assistant" | "user";
    text: string;
}

export function Client() {
    const { videoRef, canvasRef, isGrabbing, swipeDirection } =
        useHandTracking();

    const [room] = useState(new Room());
    const [isMuted, setIsMuted] = useState(true);

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
                    ctx.drawImage(videoRef.current, 0, 0);

                    // Convert canvas to blob
                    const blob = await new Promise<Blob>((resolve) => {
                        canvas.toBlob(
                            (blob) => {
                                if (blob) resolve(blob);
                            },
                            "image/jpeg",
                            0.5
                        );
                    });

                    // Create File object from blob
                    const file = new File([blob], "calibration.jpg", {
                        type: "image/jpeg",
                    });

                    const { success } = await takeCalibrationImage(file, {
                        category: "Upper-body",
                    });

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

    const toggleMute = async () => {
        const newMuteState = !isMuted;
        await room.localParticipant.setMicrophoneEnabled(!newMuteState);
        setIsMuted(newMuteState);
    };

    return (
        <div className="lk-room-container relative mx-auto h-full max-h-full w-full max-w-full overflow-hidden">
            <RoomContext.Provider value={room}>
                <div className="flex h-full flex-col items-center justify-center p-[5%]">
                    <video
                        ref={videoRef}
                        className="hidden"
                        playsInline
                    />
                    <div className="relative flex h-full min-h-full w-full min-w-full items-center justify-center overflow-hidden rounded-2xl">
                        <canvas
                            ref={canvasRef}
                            className="h-[100vw] scale-x-[-1] rotate-90 object-cover"
                        />
                    </div>

                    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-5xl">
                        {isMuted ? "Muted" : "Unmuted"}
                    </div>

                    <div className="absolute right-4 bottom-4 -translate-x-1/2">
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={toggleMute}
                            className="z-20 h-fit w-fit bg-transparent text-black"
                        >
                            {isMuted ? (
                                <MicOffIcon className="size-20" />
                            ) : (
                                <Mic2Icon className="size-20" />
                            )}
                        </Button>
                    </div>

                    {/* <TranscriptionView /> */}
                    <Captions
                        messages={messages}
                        userIsFinal={userIsFinal}
                    />
                </div>
                <RoomAudioRenderer />
            </RoomContext.Provider>
        </div>
    );
}
