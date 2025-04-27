"use client";

import { useEffect, useState } from "react";
import {
    generateClothing,
    generateTryOn,
    takeCalibrationImage,
} from "@/actions/action";
import { ConnectionDetails } from "@/app/api/connection-details/route";
import { Captions } from "@/components/captions";
import { TranscriptionView } from "@/components/livekit/transcription-view";
import { Button } from "@/components/ui/button";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog";
import { MotionImage } from "@/components/ui/motion-image";
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
import { Mic2Icon, MicOffIcon, XIcon } from "lucide-react";

// Message type
interface Message {
    role: "assistant" | "user";
    text: string;
}

export function Client() {
    const { videoRef, canvasRef, isGrabbing, swipeDirection } =
        useHandTracking();

    const [room] = useState(new Room());
    const [tryOnResult, setTryOnResult] = useState<string | null>(null);
    const [showModal, setShowModal] = useState(false);
    const [isMuted, setIsMuted] = useState(false);
    const [showClothingOptionsView, setShowClothingOptionsView] =
        useState(false);

    // New state for captions
    const [messages, setMessages] = useState<Message[]>([]);
    const [userIsFinal, setUserIsFinal] = useState<boolean>(true);

    const [garment, setGarment] = useState<File | null>(null);

    const toggleMute = async () => {
        const newMuteState = !isMuted;
        setIsMuted(newMuteState);
        await room.localParticipant.setMicrophoneEnabled(!newMuteState);
    };

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
            await room.localParticipant.setMicrophoneEnabled(!isMuted);
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
                    // Swap width and height since we're rotating 90 degrees
                    canvas.width = videoRef.current.videoHeight;
                    canvas.height = videoRef.current.videoWidth;

                    const ctx = canvas.getContext("2d");
                    if (!ctx) {
                        throw new RpcError(
                            1,
                            "Could not create canvas context"
                        );
                    }

                    // Rotate 90 degrees counterclockwise
                    ctx.translate(0, canvas.height);
                    ctx.rotate(-Math.PI / 2);
                    ctx.drawImage(videoRef.current, 0, 0);

                    // Convert canvas to blob with scaled dimensions
                    const blob = await new Promise<Blob>((resolve) => {
                        const scaledCanvas = document.createElement("canvas");
                        // Calculate width to maintain aspect ratio with height of 1024
                        const aspectRatio = canvas.width / canvas.height;
                        scaledCanvas.width = Math.round(1024 * aspectRatio);
                        scaledCanvas.height = 1024;
                        const scaledCtx = scaledCanvas.getContext("2d");
                        if (scaledCtx) {
                            scaledCtx.drawImage(
                                canvas,
                                0,
                                0,
                                scaledCanvas.width,
                                scaledCanvas.height
                            );
                            scaledCanvas.toBlob(
                                (blob) => {
                                    if (blob) resolve(blob);
                                },
                                "image/jpeg",
                                0.5
                            );
                        }
                    });

                    // Create File object from blob
                    const file = new File([blob], "image.jpg", {
                        type: "image/jpeg",
                    });

                    // const response = await fetch("/dev/kevin.jpg");
                    // const blob = await response.blob();
                    // const file = new File([blob], "kevin.jpg", {
                    //     type: "image/jpeg",
                    // });

                    const { status } = await takeCalibrationImage(file);

                    return status.toString();
                } catch (error) {
                    throw new RpcError(1, "Failed to capture video frame");
                }
            }
        );

        localParticipant.registerRpcMethod(
            "generateClothing",
            async (data: RpcInvocationData) => {
                try {
                    if (!videoRef.current) {
                        throw new RpcError(1, "Video element not found");
                    }

                    const canvas = document.createElement("canvas");
                    // Swap width and height since we're rotating 90 degrees
                    canvas.width = videoRef.current.videoHeight;
                    canvas.height = videoRef.current.videoWidth;

                    const ctx = canvas.getContext("2d");
                    if (!ctx) {
                        throw new RpcError(
                            1,
                            "Could not create canvas context"
                        );
                    }

                    // Rotate 90 degrees counterclockwise
                    ctx.translate(0, canvas.height);
                    ctx.rotate(-Math.PI / 2);
                    ctx.drawImage(videoRef.current, 0, 0);

                    // Convert canvas to blob with scaled dimensions
                    const vtonBlob = await new Promise<Blob>((resolve) => {
                        const scaledCanvas = document.createElement("canvas");
                        // Calculate width to maintain aspect ratio with height of 1024
                        const aspectRatio = canvas.width / canvas.height;
                        scaledCanvas.width = Math.round(1024 * aspectRatio);
                        scaledCanvas.height = 1024;
                        const scaledCtx = scaledCanvas.getContext("2d");
                        if (scaledCtx) {
                            scaledCtx.drawImage(
                                canvas,
                                0,
                                0,
                                scaledCanvas.width,
                                scaledCanvas.height
                            );
                            scaledCanvas.toBlob(
                                (blob) => {
                                    if (blob) resolve(blob);
                                },
                                "image/jpeg",
                                0.5
                            );
                        }
                    });

                    // Create File object from blob
                    const vtonFile = new File([vtonBlob], "image.jpg", {
                        type: "image/jpeg",
                    });

                    const response = await fetch("/dev/bunny.jpg");
                    const blob = await response.blob();

                    // Create File object from blob
                    const garmentFile = new File([blob], "garment.jpg", {
                        type: "image/jpeg",
                    });

                    const { result } = await generateClothing(
                        garmentFile,
                        "Upper-body",
                        vtonFile
                    );
                    console.log(result.toString());
                    console.log("SUCCESS");

                    // Store the result and show modal
                    setTryOnResult(result.toString());
                    setShowModal(true);

                    return "SUCCESS";
                } catch (error) {
                    throw new RpcError(1, "Failed to capture video frame");
                }
            }
        );

        localParticipant.registerRpcMethod(
            "showClothingOptions",
            async (data: RpcInvocationData) => {
                const payload = JSON.parse(data.payload);

                setShowClothingOptionsView(payload.show);

                return "SUCCESS";
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

                    <div className="absolute top-8 left-8 -translate-x-1/2">
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={toggleMute}
                            className="z-20 h-fit w-fit bg-transparent text-black"
                        >
                            {isMuted ? (
                                <MicOffIcon className="size-24" />
                            ) : (
                                <Mic2Icon className="size-24" />
                            )}
                        </Button>
                    </div>

                    {/* <TranscriptionView /> */}
                    <Captions
                        messages={messages}
                        userIsFinal={userIsFinal}
                    />

                    {showClothingOptionsView && (
                        <MotionImage
                            images={[
                                {
                                    src: "/dev/bunny.jpg",
                                    alt: "Bunny",
                                    width: 1024,
                                    height: 1024,
                                },
                                {
                                    src: "/dev/kevin.jpg",
                                    alt: "Kevin",
                                    width: 1024,
                                    height: 1024,
                                },
                                {
                                    src: "/dev/garment.jpg",
                                    alt: "Dress",
                                    width: 1024,
                                    height: 1024,
                                },
                            ]}
                        />
                    )}
                </div>
                <RoomAudioRenderer />
            </RoomContext.Provider>

            <Dialog
                open={showModal}
                onOpenChange={setShowModal}
            >
                <DialogContent className="max-w-none sm:max-w-[calc(100vw-32rem)]">
                    <img
                        src={`data:image/jpeg;base64,${tryOnResult}`}
                        alt="Try-on Result"
                        className="h-[80vh]"
                    />
                </DialogContent>
            </Dialog>
        </div>
    );
}
