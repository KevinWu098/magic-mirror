"use client";

import Link from "next/link";
import { TextHoverEffect } from "@/components/aceternity/text-hover-effect";
import { motion } from "motion/react";

export default function Page() {
    return (
        <Link href="/">
            <div className="flex h-full min-h-screen cursor-pointer flex-col justify-end bg-[url('/dev/ward.png')] bg-cover bg-center bg-no-repeat p-32 text-center">
                <div className="absolute inset-0 bg-black/40"></div>

                <div className="relative z-10 space-y-12">
                    <h1 className="font-birthstone text-[36rem] leading-[0.75] text-white">
                        Magic <br /> Mirror
                    </h1>
                    {/* <div className="-space-y-64">
                        <TextHoverEffect text="Magic" />
                        <TextHoverEffect text="Mirror" />
                    </div> */}

                    <motion.h2
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.5 }}
                        className="font-sans text-7xl text-balance text-white"
                    >
                        The world's first interactive, virtual try-on studio
                    </motion.h2>
                </div>
            </div>
        </Link>
    );
}
