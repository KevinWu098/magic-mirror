"use client";

import { useMemo, useState, useEffect } from "react";
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

            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            
            // Mark pixels to be transparent
            const visited = new Array(canvas.width * canvas.height).fill(false);
            const queue: [number, number][] = [[0, 0]]; // Start BFS from (0,0)
            
            // Threshold for what we consider "white" - adjust as needed
            const isWhite = (r: number, g: number, b: number) => r > 240 && g > 240 && b > 240;
            
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
                "relative flex flex-col items-center rounded-3xl bg-white/20 backdrop-blur-lg border border-white/30 shadow-lg overflow-hidden p-6 min-w-[280px] max-w-[340px] transition-all",
                className
            )}
            style={{ boxShadow: "0 8px 24px rgba(0, 0, 0, 0.1)" }}
        >
            {/* Product Image with Fixed Size and Uniform Scaling */}
            <div className="relative w-full h-[180px] flex items-center justify-center">
                <div className="relative w-[180px] h-[180px]">
                    {processedImageUrl && (
                        <Image
                            src={processedImageUrl}
                            alt={name}
                            fill
                            className="object-contain"
                            sizes="180px"
                            priority
                        />
                    )}
                </div>
            </div>

            {/* Product Info */}
            <div className="flex flex-col items-center text-center mt-4">
                <span className="text-2xl font-bold text-gray-900">{formattedPrice}</span>
                <span className="text-lg font-semibold text-gray-800 mt-1 line-clamp-2">{name}</span>
            </div>

            {/* Website Info at Bottom */}
            <div className="flex items-center gap-2 mt-6">
                <div className="relative h-6 w-6">
                    <Image
                        src={websiteIcon}
                        alt="Website"
                        fill
                        className="object-contain rounded-full"
                    />
                </div>
                <Link
                    href={websiteUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-600 hover:text-gray-800 text-sm font-medium"
                >
                    {new URL(websiteUrl).hostname.replace("www.", "")}
                </Link>
            </div>
        </div>
    );
};

export default ProductBubble;
