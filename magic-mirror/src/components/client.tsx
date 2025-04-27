"use client";

import { useEffect, useRef, useState } from "react";
import {
    generateClothing,
    generateTryOn,
    takeCalibrationImage,
} from "@/actions/action";
import { generateImage } from "@/actions/image";
import { findSimilarClothing } from "@/actions/search";
import { ConnectionDetails } from "@/app/api/connection-details/route";
import { Captions } from "@/components/captions";
import { ClothingDialog } from "@/components/clothing-dialog";
import { GeneratedClothingDialog } from "@/components/generated-clothing-dialog";
import { TranscriptionView } from "@/components/livekit/transcription-view";
import { ProductBubble } from "@/components/product-bubble";
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
import { captureAndProcessVideoFrame } from "@/lib/rpc";
import { cn } from "@/lib/utils";
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
import {
    DownloadIcon,
    Mic2Icon,
    MicOffIcon,
    ShirtIcon,
    XIcon,
} from "lucide-react";
import { motion } from "motion/react";
import { toast } from "sonner";

// Message type
interface Message {
    role: "assistant" | "user";
    text: string;
}

interface ImageItem {
    src: string;
    alt: string;
    width: number;
    height: number;
}

interface SimilarClothingResult {
    status: string;
    request_id: string;
    parameters: {
        url: string;
        language: string;
        country: string;
    };
    data: {
        visual_matches: Array<{
            position: number;
            title: string;
            link: string;
            source: string;
            source_icon: string;
            thumbnail: string;
            thumbnail_width: number;
            thumbnail_height: number;
            image: string;
            image_width: number;
            image_height: number;
            price: string;
            availability: string;
        }>;
    };
}

interface SimilarClothingItem {
    imageUrl: string;
    price: number;
    name: string;
    websiteUrl: string;
    websiteIcon: string;
}

const SHOULD_CONNECT = true;

export function Client() {
    const { videoRef, canvasRef, isGrabbing, swipeDirection } =
        useHandTracking();

    const [room] = useState(new Room());
    const [tryOnResult, setTryOnResult] = useState<string | null>(null);
    const [tryOnOriginalResult, setTryOnOriginalResult] = useState<
        string | null
    >(null);
    const [generatedImage1, setGeneratedImage1] = useState<string | null>(null);
    const [generatedImage2, setGeneratedImage2] = useState<string | null>(null);

    const [showModal, setShowModal] = useState(false);
    const [showGeneratedModal, setShowGeneratedModal] = useState(false);

    const [isMuted, setIsMuted] = useState(false);
    const [isVideoLoaded, setIsVideoLoaded] = useState(false);

    const [showClothingOptionsView, setShowClothingOptionsView] =
        useState(false);
    const [showSearchOptionsView, setShowSearchOptionsView] = useState(false);

    // New state for captions
    const [messages, setMessages] = useState<Message[]>([]);
    const [userIsFinal, setUserIsFinal] = useState<boolean>(true);

    const [garment, setGarment] = useState<File | null>(null);

    const [foobar, setFoobar] = useState<File[]>([]);
    const [images, setImages] = useState<ImageItem[]>([
        {
            src: "/dev/bunny.jpg",
            alt: "Bunny",
            width: 1024,
            height: 1024,
        },
        {
            src: "/dev/garment.jpg",
            alt: "Dress",
            width: 1024,
            height: 1024,
        },
    ]);

    const [similarClothingItems, setSimilarClothingItems] = useState<
        SimilarClothingItem[]
    >([]);

    const toggleMute = async () => {
        const newMuteState = !isMuted;
        setIsMuted(newMuteState);
        await room.localParticipant.setMicrophoneEnabled(!newMuteState);
    };

    const [loading, setLoading] = useState(false);
    const [isDragging, setIsDragging] = useState(false);

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

            // Get available audio input devices
            const devices = await Room.getLocalDevices("audioinput");
            console.log("devices", devices);
            if (devices.length > 1) {
                // Use the second microphone if available
                await room.localParticipant.setMicrophoneEnabled(!isMuted, {
                    deviceId: devices[0]?.deviceId,
                });
            } else {
                // Fallback to default microphone
                await room.localParticipant.setMicrophoneEnabled(!isMuted);
            }
        }

        if (SHOULD_CONNECT) {
            connect();
        }

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
            "tryOnClothing",
            async (data: RpcInvocationData) => {
                try {
                    const payload = JSON.parse(data.payload);
                    const bodyPart = payload.body_part;

                    setLoading(true);
                    setShowModal(false);
                    setShowGeneratedModal(false);

                    const vtonBlob =
                        await captureAndProcessVideoFrame(videoRef);

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

                    setGarment(garmentFile);
                    setFoobar([garmentFile]);

                    const { maskedImage, overlaidImage, originalImage } =
                        await generateClothing(garmentFile, bodyPart, vtonFile);
                    console.log("SUCCESS");

                    // Store the result and show modal
                    setTryOnResult(overlaidImage.toString());
                    setTryOnOriginalResult(originalImage.toString());
                    setShowModal(true);

                    return "SUCCESS";
                } catch (error) {
                    throw new RpcError(1, "Failed to capture video frame");
                } finally {
                    setLoading(false);
                }
            }
        );

        localParticipant.registerRpcMethod(
            "tryOnCreativeClothing",
            async (data: RpcInvocationData) => {
                try {
                    setLoading(true);
                    setShowModal(false);
                    setShowGeneratedModal(false);

                    const payload = JSON.parse(data.payload);
                    const generationRequest = payload.generationRequest;
                    const bodyPart = payload.body_part;
                    setGeneratedImage1(null);
                    setGeneratedImage2(null);

                    const vtonBlob =
                        await captureAndProcessVideoFrame(videoRef);

                    // setShowGeneratedModal(true);

                    // Generate images independently to handle them as they complete
                    const result = await generateImage(
                        null,
                        generationRequest,
                        "low"
                    );
                    const canvas = document.createElement("canvas");
                    canvas.width = 768;
                    canvas.height = 1024;
                    const ctx = canvas.getContext("2d");

                    const img = new Image();
                    img.src = `data:image/jpeg;base64,${result.imageBase64}`;
                    await new Promise((resolve) => {
                        img.onload = resolve;
                    });

                    ctx?.drawImage(img, 0, 0, 768, 1024);
                    const resizedBase64 = canvas
                        .toDataURL("image/jpeg")
                        .split(",")[1]!;
                    result.imageBase64 = resizedBase64;
                    setGeneratedImage1(result.imageBase64);

                    setShowGeneratedModal(true);

                    // Create File object from blob
                    const vtonFile = new File([vtonBlob], "image.jpg", {
                        type: "image/jpeg",
                    });

                    const blob = await fetch(
                        `data:image/jpeg;base64,${resizedBase64}`
                    ).then((res) => res.blob());

                    // Create File object from blob
                    const garmentFile = new File([blob], "garment.jpg", {
                        type: "image/jpeg",
                    });

                    setGarment(garmentFile);
                    setFoobar([garmentFile]);

                    const { maskedImage, overlaidImage, originalImage } =
                        await generateClothing(garmentFile, bodyPart, vtonFile);
                    console.log("SUCCESS");

                    // Store the result and show modal
                    setTryOnResult(overlaidImage.toString());
                    setTryOnOriginalResult(originalImage.toString());
                    setShowGeneratedModal(false);
                    setShowModal(true);

                    return "SUCCESS";
                } catch (error) {
                    throw new RpcError(1, "Failed to capture video frame");
                } finally {
                    setLoading(false);
                }
            }
        );

        localParticipant.registerRpcMethod(
            "showClothingOptions",
            async (data: RpcInvocationData) => {
                const payload = JSON.parse(data.payload);

                setShowSearchOptionsView(false);
                setShowClothingOptionsView(payload.show);

                return "SUCCESS";
            }
        );

        localParticipant.registerRpcMethod(
            "showStandardModal",
            async (data: RpcInvocationData) => {
                const payload = JSON.parse(data.payload);

                setShowGeneratedModal(false);
                setShowModal(payload.show);
                setShowClothingOptionsView(false);
                setShowSearchOptionsView(false);

                return "SUCCESS";
            }
        );

        localParticipant.registerRpcMethod(
            "showCreativeModal",
            async (data: RpcInvocationData) => {
                const payload = JSON.parse(data.payload);

                setShowModal(false);
                setShowGeneratedModal(payload.show);
                setShowClothingOptionsView(false);
                setShowSearchOptionsView(false);

                return "SUCCESS";
            }
        );

        localParticipant.registerRpcMethod(
            "showSearchOptions",
            async (data: RpcInvocationData) => {
                const payload = JSON.parse(data.payload);

                setShowClothingOptionsView(false);
                setShowSearchOptionsView(payload.show);

                return "SUCCESS";
            }
        );

        localParticipant.registerRpcMethod(
            "findSimilarClothing",
            async (data: RpcInvocationData) => {
                console.log("GARMENT", garment);
                console.log("GARMENT", foobar);

                setLoading(true);
                setShowSearchOptionsView(false);
                setShowClothingOptionsView(false);

                const garmentToUse =
                    foobar[0] ||
                    garment ||
                    new File(
                        [await fetch(images[0]!.src).then((r) => r.blob())],
                        "garment.jpg",
                        {
                            type: "image/jpeg",
                        }
                    );

                console.log("garmenttouse", garmentToUse);

                const result: SimilarClothingResult =
                    await findSimilarClothing(garmentToUse);

                console.log(result);

                const validProducts = result.data.visual_matches
                    .filter((e) => e.price && e.availability === "In stock")
                    .slice(0, 3);

                // Transform the products into our format
                const transformedProducts: SimilarClothingItem[] =
                    validProducts.map((product) => ({
                        imageUrl: product.image,
                        price: parseFloat(
                            product.price.replace(/[^0-9.]/g, "")
                        ),
                        name: product.title,
                        websiteUrl: product.link,
                        websiteIcon: product.source_icon,
                    }));

                console.log(transformedProducts);

                setSimilarClothingItems(transformedProducts);
                setShowSearchOptionsView(true);
                setShowModal(false);
                setShowGeneratedModal(false);

                setLoading(false);

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

    useEffect(() => {
        const handleVideoLoad = () => {
            if (videoRef.current) {
                setIsVideoLoaded(true);
            }
        };

        if (videoRef.current) {
            videoRef.current.addEventListener("loadeddata", handleVideoLoad);
        }

        return () => {
            if (videoRef.current) {
                videoRef.current.removeEventListener(
                    "loadeddata",
                    handleVideoLoad
                );
            }
        };
    }, [videoRef]);

    // Add drag and drop handlers
    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    };

    const handleDrop = async (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        // Add detailed logging
        console.log("Drop event:", e);
        console.log("Files:", Array.from(e.dataTransfer.files));
        console.log("Types:", e.dataTransfer.types);

        // Handle images dragged from web
        const items = Array.from(e.dataTransfer.items);
        console.log(
            "Items:",
            items.map((item) => ({
                kind: item.kind,
                type: item.type,
            }))
        );

        for (const item of items) {
            console.log("Processing item:", {
                kind: item.kind,
                type: item.type,
            });

            // Try to get image data directly
            if (item.type.startsWith("image/")) {
                console.log("Found image type");
                const file = item.getAsFile();
                console.log("File from image:", file);
                if (file) {
                    const reader = new FileReader();
                    reader.onload = (event) => {
                        if (!event.target?.result) {
                            console.error("Failed to read file");
                            return;
                        }

                        const base64String = event.target.result as string;
                        console.log("Successfully read image data");
                        const newImage: ImageItem = {
                            src: base64String,
                            alt: "Dragged image",
                            width: 1024,
                            height: 1024,
                        };
                        setImages((prevImages) => [newImage, ...prevImages]);
                        setGarment(file);
                    };
                    reader.readAsDataURL(file);
                }
                continue;
            }

            // Fallback to handling URL if available
            if (item.kind === "string") {
                if (
                    item.type === "text/uri-list" ||
                    item.type === "text/plain" ||
                    item.type === "text/html"
                ) {
                    console.log("Found string type:", item.type);
                    item.getAsString((data) => {
                        console.log("String data:", data);
                        // Try to extract image URL from HTML if present
                        let url = data;

                        // First try to extract from HTML img tag
                        const imgSrcMatch = data.match(
                            /img[^>]+src=["']([^"']+)/i
                        );
                        if (imgSrcMatch && imgSrcMatch[1]) {
                            url = imgSrcMatch[1];
                            console.log("Found img src URL:", url);
                        }
                        // If no img tag or if it's a direct URL
                        if (
                            !url.includes("img") &&
                            url.match(/^https?:\/\/[^\s<>"']+/i)
                        ) {
                            console.log("Using direct URL:", url);
                        }

                        // If we have a URL that looks like an image URL, use it
                        if (url.match(/\.(jpeg|jpg|gif|png|webp)/i) !== null) {
                            console.log("Found valid image URL:", url);
                            const newImage: ImageItem = {
                                src: url,
                                alt: "Dragged image",
                                width: 1024,
                                height: 1024,
                            };
                            setImages((prevImages) => [
                                newImage,
                                ...prevImages,
                            ]);
                            toast.success("Clothing uploaded successfully", {
                                position: "top-center",
                            });
                        } else {
                            console.log("No valid image URL found in:", url);
                        }
                    });
                }
            }
        }
    };

    console.log(garment);
    console.log(showSearchOptionsView);

    return (
        <div
            className={cn(
                "lk-room-container relative mx-auto h-full max-h-full w-full max-w-full overflow-hidden"
            )}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
            <RoomContext.Provider value={room}>
                {/* <div className="text-5xl">Magic Mirror</div> */}
                <div className="flex h-full flex-col items-center justify-center p-[5%] pb-[7%]">
                    <video
                        ref={videoRef}
                        className="hidden"
                        playsInline
                    />

                    <div className="relative flex h-full min-h-full w-full min-w-full items-center justify-center overflow-hidden rounded-2xl">
                        <motion.canvas
                            ref={canvasRef}
                            className="pointer-events-none h-[100vw] scale-x-[-1] rotate-90 object-cover"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: isVideoLoaded ? 1 : 0 }}
                            transition={{ duration: 0.5, ease: "easeInOut" }}
                        />
                        {loading && (
                            <motion.div
                                animate={{ opacity: isVideoLoaded ? 1 : 0 }}
                                className="pointer-events-none absolute inset-0 h-full w-full animate-pulse border-[24px] border-blue-400"
                            />
                        )}
                        {isDragging && (
                            <motion.div
                                animate={{ opacity: isVideoLoaded ? 1 : 0 }}
                                className="pointer-events-none absolute inset-0 h-full w-full animate-pulse border-[24px] border-green-400"
                            />
                        )}
                    </div>

                    {(!canvasRef.current || !isVideoLoaded) && (
                        <div className="relative flex h-full min-h-full w-full min-w-full -translate-y-1/2 items-center justify-center overflow-hidden rounded-2xl">
                            <div className="pointer-events-none absolute inset-0 z-10 animate-pulse rounded-2xl bg-gray-300 dark:bg-gray-700" />
                        </div>
                    )}

                    <div className="absolute bottom-4 left-1/2 z-20 mx-auto grid w-full -translate-x-1/2 grid-cols-3 items-center justify-center gap-4">
                        {images.length > 2 && images[0]?.src ? (
                            <Button
                                variant="ghost"
                                size="icon"
                                className="mx-auto h-fit w-fit bg-transparent text-black"
                                onClick={() => {
                                    const imageUrl = images[0]!.src;
                                    // Download the most recent image
                                    const link = document.createElement("a");
                                    link.href = imageUrl;
                                    link.download = "dragged-image.jpg";
                                    document.body.appendChild(link);
                                    link.click();
                                    document.body.removeChild(link);
                                    toast.success(
                                        "Image downloaded successfully"
                                    );
                                }}
                            >
                                <DownloadIcon className="size-16 fill-green-400 text-green-400" />
                            </Button>
                        ) : (
                            <div className="size-16" /> // Empty placeholder to maintain grid spacing
                        )}

                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={toggleMute}
                            className="mx-auto h-fit w-fit bg-transparent text-black"
                        >
                            {isMuted ? (
                                <MicOffIcon className="size-16" />
                            ) : (
                                <Mic2Icon className="size-16" />
                            )}
                        </Button>

                        <Button
                            variant="ghost"
                            size="icon"
                            className={cn(
                                "mx-auto h-fit w-fit bg-transparent text-black",
                                !garment && "invisible"
                            )}
                        >
                            <ShirtIcon className="size-16 fill-green-400 text-green-400" />
                        </Button>
                    </div>

                    {/* <div className="absolute top-8 left-8 -translate-x-1/2">
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
                    </div> */}

                    {/* <TranscriptionView /> */}
                    <Captions
                        messages={messages}
                        userIsFinal={userIsFinal}
                    />

                    {showClothingOptionsView && <MotionImage images={images} />}
                    {showSearchOptionsView && (
                        <div className="absolute bottom-32 grid w-full grid-cols-3 gap-8 px-40">
                            {similarClothingItems.map((item, index) => (
                                <motion.div
                                    key={index}
                                    initial={{ opacity: 0, y: 50 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: 50 }}
                                    transition={{
                                        duration: 1,
                                        delay: index * 0.2,
                                        ease: [0.32, 0.72, 0, 1],
                                    }}
                                >
                                    <ProductBubble
                                        imageUrl={item.imageUrl}
                                        price={item.price}
                                        name={item.name}
                                        websiteUrl={item.websiteUrl}
                                        websiteIcon={item.websiteIcon}
                                    />
                                </motion.div>
                            ))}
                        </div>
                    )}
                </div>
                <RoomAudioRenderer />
            </RoomContext.Provider>

            <ClothingDialog
                showModal={showModal}
                setShowModal={setShowModal}
                tryOnResult={tryOnResult}
                tryOnOriginalResult={tryOnOriginalResult}
            />

            <GeneratedClothingDialog
                showModal={showGeneratedModal}
                setShowModal={setShowGeneratedModal}
                generatedImage1={generatedImage1}
                generatedImage2={generatedImage2}
            />
        </div>
    );
}
