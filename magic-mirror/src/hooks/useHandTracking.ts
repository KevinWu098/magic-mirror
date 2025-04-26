import { useEffect, useRef, useState } from "react";
import {
    DrawingUtils,
    FilesetResolver,
    HandLandmarker,
    HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

// Constants for gesture detection
const DRAG_THRESHOLD = 30; // Distance threshold for drag detection
const FINGER_CLOSED_THRESHOLD = 0.04; // Threshold for detecting closed fingers
const SWIPE_THRESHOLD = 100; // Minimum distance for swipe detection
const SWIPE_COOLDOWN = 500; // Cooldown period between swipes in milliseconds

export function useHandTracking() {
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const landmarkerRef = useRef<HandLandmarker | null>(null);

    const [isGrabbing, setIsGrabbing] = useState(false);
    const [swipeDirection, setSwipeDirection] = useState<
        "left" | "right" | null
    >(null);
    const dragStartPosition = useRef<{ x: number; y: number } | null>(null);
    const lastSwipeTime = useRef<number>(0);

    useEffect(() => {
        init();
        return () => {
            if (videoRef.current && videoRef.current.srcObject) {
                const tracks = (
                    videoRef.current.srcObject as MediaStream
                ).getTracks();
                tracks.forEach((track) => track.stop());
            }
        };
    }, []);

    const init = async () => {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm"
        );
        const landmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath:
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                delegate: "GPU",
            },
            runningMode: "VIDEO",
            numHands: 1,
        });

        landmarkerRef.current = landmarker;

        // Set up video stream
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 720, height: 1280 },
            });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.play();
            }
            startTracking(landmarker);
        } catch (error) {
            console.error("Error accessing webcam:", error);
        }
    };

    const startTracking = (landmarker: HandLandmarker) => {
        if (!videoRef.current || !canvasRef.current || !landmarker) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const drawingUtils = new DrawingUtils(ctx);

        const processFrame = async () => {
            if (!video || !canvas || !ctx || !landmarker) return;

            // Only process frame if video is playing
            if (!video.paused && !video.ended) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                const startTimeMs = performance.now();
                const results = landmarker.detectForVideo(video, startTimeMs);

                processResults(results, canvas.width, canvas.height);
                drawResults(results, drawingUtils);
            }

            requestAnimationFrame(processFrame);
        };

        video.addEventListener("loadeddata", () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            processFrame();
        });
    };

    const processResults = (
        results: HandLandmarkerResult,
        width: number,
        height: number
    ) => {
        if (!results.landmarks || results.landmarks.length === 0) {
            setIsGrabbing(false);
            dragStartPosition.current = null;
            setSwipeDirection(null);
            return;
        }

        results.landmarks.forEach((landmarks) => {
            // Check if fingers are extended (with null checks)
            const indexExtended =
                landmarks[8]?.y !== undefined &&
                landmarks[5]?.y !== undefined &&
                landmarks[8].y < landmarks[5].y - FINGER_CLOSED_THRESHOLD;
            const middleExtended =
                landmarks[12]?.y !== undefined &&
                landmarks[9]?.y !== undefined &&
                landmarks[12].y < landmarks[9].y - FINGER_CLOSED_THRESHOLD;
            const ringExtended =
                landmarks[16]?.y !== undefined &&
                landmarks[13]?.y !== undefined &&
                landmarks[16].y < landmarks[13].y - FINGER_CLOSED_THRESHOLD;
            const pinkyExtended =
                landmarks[20]?.y !== undefined &&
                landmarks[17]?.y !== undefined &&
                landmarks[20].y < landmarks[17].y - FINGER_CLOSED_THRESHOLD;

            // Check for fist gesture
            const isFist =
                !indexExtended &&
                !middleExtended &&
                !ringExtended &&
                !pinkyExtended;

            // Get palm position with null check
            const palmLandmark = landmarks[9];
            if (!palmLandmark?.x || !palmLandmark?.y) return;

            const palmX = palmLandmark.x * width;
            const palmY = palmLandmark.y * height;

            if (isFist) {
                // Use the ref value directly instead of the state to avoid stale closures
                if (!dragStartPosition.current) {
                    setIsGrabbing(true);
                    dragStartPosition.current = { x: palmX, y: palmY };
                    setSwipeDirection(null);
                } else {
                    const currentTime = Date.now();
                    const horizontalDistance =
                        palmX - dragStartPosition.current.x;
                    const timeSinceLastSwipe =
                        currentTime - lastSwipeTime.current;

                    // Only detect swipes if enough time has passed since the last one
                    if (
                        Math.abs(horizontalDistance) > SWIPE_THRESHOLD &&
                        timeSinceLastSwipe > SWIPE_COOLDOWN
                    ) {
                        const newDirection =
                            horizontalDistance > 0 ? "right" : "left";
                        setSwipeDirection(newDirection);
                        lastSwipeTime.current = currentTime;
                        dragStartPosition.current = { x: palmX, y: palmY }; // Reset start position after swipe
                    }
                }
            } else {
                setIsGrabbing(false);
                dragStartPosition.current = null;
                setSwipeDirection(null);
            }
        });
    };

    const drawResults = (
        results: HandLandmarkerResult,
        drawingUtils: DrawingUtils
    ) => {
        if (!results.landmarks || !results.worldLandmarks) return;

        for (const landmarks of results.landmarks) {
            drawingUtils.drawConnectors(
                landmarks,
                HandLandmarker.HAND_CONNECTIONS,
                { color: "#00FF00", lineWidth: 2 }
            );
            drawingUtils.drawLandmarks(landmarks, {
                color: "#FF0000",
                lineWidth: 1,
                radius: 3,
            });
        }
    };

    return {
        videoRef,
        canvasRef,
        isGrabbing,
        swipeDirection,
    };
}
