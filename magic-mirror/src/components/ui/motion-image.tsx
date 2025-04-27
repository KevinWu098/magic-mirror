"use client";

import Image from "next/image";
import { cn } from "@/lib/utils";
import { motion } from "motion/react";

interface MotionImageProps {
    containerClassName?: string;
    className?: string;
    images: {
        src: string;
        alt: string;
        width: number;
        height: number;
    }[];
}

export function MotionImage({
    containerClassName,
    className,
    images,
}: MotionImageProps) {
    const motionDivs = images.map((image, index) => (
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
            className={cn(
                "aspect-square w-1/3 overflow-hidden rounded-xl border-8 border-blue-500",
                containerClassName
            )}
        >
            <Image
                className={cn(
                    "pointer-events-none aspect-square w-full object-cover object-top",
                    className
                )}
                {...image}
            />
        </motion.div>
    ));

    return (
        <div className="absolute bottom-32 flex w-full items-center justify-between gap-8 px-40">
            {motionDivs}
        </div>
    );
}
