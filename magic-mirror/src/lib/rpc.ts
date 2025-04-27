import { RpcError } from "livekit-client";

export async function captureAndProcessVideoFrame(
    videoRef: React.RefObject<HTMLVideoElement | null>
) {
    if (!videoRef.current) {
        throw new RpcError(1, "Video element not found");
    }

    const canvas = document.createElement("canvas");
    // Swap width and height since we're rotating 90 degrees
    canvas.width = videoRef.current.videoHeight;
    canvas.height = videoRef.current.videoWidth;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
        throw new RpcError(1, "Could not create canvas context");
    }

    // Rotate 90 degrees counterclockwise
    ctx.translate(0, canvas.height);
    ctx.rotate(-Math.PI / 2);
    ctx.drawImage(videoRef.current, 0, 0);

    // Convert canvas to blob with scaled dimensions and mirror the image
    const vtonBlob = await new Promise<Blob>((resolve) => {
        const scaledCanvas = document.createElement("canvas");
        // Calculate width to maintain aspect ratio with height of 1024
        const aspectRatio = canvas.width / canvas.height;
        scaledCanvas.width = Math.round(1024 * aspectRatio);
        scaledCanvas.height = 1024;
        const scaledCtx = scaledCanvas.getContext("2d");
        if (scaledCtx) {
            // Mirror the image horizontally
            scaledCtx.translate(scaledCanvas.width, 0);
            scaledCtx.scale(-1, 1);
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

    return vtonBlob;
}
