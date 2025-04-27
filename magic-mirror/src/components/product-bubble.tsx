"use client";

import { useEffect, useMemo, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { cn } from "@/lib/utils";

interface ProductBubbleProps {
    imageUrl: string;
    price: number;
    name: string;
    websiteUrl: string;
    websiteIcon: string;
    className?: string;
}

// BFS function to remove white background pixels starting from (0,0)
const removeWhiteBackground = (imageUrl: string): Promise<string> => {
    // Check if running on the browser
    if (typeof window === "undefined") {
        return Promise.resolve(imageUrl);
    }

    return new Promise((resolve) => {
        const img = new globalThis.Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            if (!ctx) {
                resolve(imageUrl); // Fallback to original if no context
                return;
            }

            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            const imageData = ctx.getImageData(
                0,
                0,
                canvas.width,
                canvas.height
            );
            const data = imageData.data;

            // Mark pixels to be transparent
            const visited = new Array(canvas.width * canvas.height).fill(false);
            const queue: [number, number][] = [[0, 0]]; // Start BFS from (0,0)

            // Threshold for what we consider "white" - adjust as needed
            const isWhite = (r: number, g: number, b: number) =>
                r > 240 && g > 240 && b > 240;

            // BFS to find connected white pixels
            while (queue.length > 0) {
                const [x, y] = queue.shift()!;
                const idx = (y * canvas.width + x) * 4;

                // Skip if already visited or out of bounds
                if (
                    visited[y * canvas.width + x] ||
                    x < 0 ||
                    y < 0 ||
                    x >= canvas.width ||
                    y >= canvas.height
                ) {
                    continue;
                }

                // Mark as visited
                visited[y * canvas.width + x] = true;

                // Check if pixel is white
                const r = data[idx] || 0;
                const g = data[idx + 1] || 0;
                const b = data[idx + 2] || 0;

                if (isWhite(r, g, b)) {
                    // Make pixel transparent
                    data[idx + 3] = 0;

                    // Add neighboring pixels to queue
                    queue.push([x + 1, y]); // right
                    queue.push([x - 1, y]); // left
                    queue.push([x, y + 1]); // down
                    queue.push([x, y - 1]); // up
                }
            }

            // Update canvas with the modified image data
            ctx.putImageData(imageData, 0, 0);
            resolve(canvas.toDataURL());
        };

        img.onerror = () => {
            resolve(imageUrl); // Fallback to original on error
        };

        img.src = imageUrl;
    });
};

export const ProductBubble = ({
    imageUrl,
    price,
    name,
    websiteUrl,
    websiteIcon,
    className,
}: ProductBubbleProps) => {
    const formattedPrice = useMemo(() => {
        return new Intl.NumberFormat("en-US", {
            style: "currency",
            currency: "USD",
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
        }).format(price);
    }, [price]);

    // State to hold the processed image URL
    const [processedImageUrl, setProcessedImageUrl] = useState("");

    // Process image when component mounts or imageUrl changes
    useEffect(() => {
        // Skip on server-side rendering
        if (typeof window === "undefined") return;

        const processImage = async () => {
            try {
                const result = await removeWhiteBackground(imageUrl);
                setProcessedImageUrl(result);
            } catch (error) {
                console.error("Error processing image:", error);
                // Keep original image on error
            }
        };

        processImage();
    }, [imageUrl]);

    return (
        <div
            className={cn(
                "relative flex max-w-[340px] min-w-[280px] flex-col items-center overflow-hidden rounded-3xl border border-white/30 bg-white/20 p-6 shadow-lg backdrop-blur-lg transition-all",
                className
            )}
            style={{ boxShadow: "0 8px 24px rgba(0, 0, 0, 0.1)" }}
        >
            {/* Product Image with Fixed Size and Uniform Scaling */}
            <div className="relative flex h-[300px] w-full items-center justify-center rounded-lg">
                <div className="relative h-full w-full rounded-lg">
                    {processedImageUrl && (
                        <Image
                            src={processedImageUrl}
                            alt={name}
                            fill
                            className="rounded-lg object-contain"
                            sizes="220px"
                            priority
                        />
                    )}
                </div>

                {/* Price Bubble */}
                <div className="absolute bottom-0 left-1/2 -translate-x-1/2 transform rounded-lg bg-black/80 px-3 py-1 text-6xl font-bold text-emerald-400 shadow-md">
                    {formattedPrice}
                </div>
            </div>
            {/* Product Info
            <div className="mt-1 flex flex-col items-center text-center">
                <span className="line-clamp-2 text-lg font-semibold text-gray-800">
                    {name}
                </span>
            </div> */}

            {/* Website Info at Bottom */}
            <div className="mt-6 flex scale-[2.25] items-center gap-2">
                <div className="relative h-6 w-6">
                    <Image
                        src={websiteIcon}
                        alt="Website"
                        fill
                        className="rounded-full object-contain"
                    />
                </div>
                <Link
                    href={websiteUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-medium text-gray-600 hover:text-gray-800"
                >
                    {new URL(websiteUrl).hostname.replace("www.", "")}
                </Link>
            </div>
        </div>
    );
};

export default ProductBubble;
