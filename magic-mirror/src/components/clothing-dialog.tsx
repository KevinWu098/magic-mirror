import { Dialog, DialogContent } from "@/components/ui/dialog";
import { motion } from "motion/react";

interface ClothingDialogProps {
    showModal: boolean;
    setShowModal: (show: boolean) => void;
    tryOnResult: string | null;
    tryOnOriginalResult: string | null;
}

export function ClothingDialog({
    showModal,
    setShowModal,
    tryOnResult,
    tryOnOriginalResult,
}: ClothingDialogProps) {
    return (
        <Dialog
            open={showModal}
            onOpenChange={setShowModal}
        >
            <DialogContent className="h-fit max-w-none p-0 sm:max-w-[calc(100vw-16rem)]">
                <div className="relative min-h-[82.25vh]">
                    <motion.img
                        initial={{ opacity: 1 }}
                        animate={{ opacity: 1 }}
                        transition={{
                            delay: 1,
                            duration: 2.5,
                            ease: "easeInOut",
                        }}
                        src={`data:image/jpeg;base64,${tryOnResult}`}
                        alt="Try-on Result"
                        className="absolute inset-0 h-full w-full object-contain"
                    />
                    <motion.img
                        initial={{ opacity: 1 }}
                        animate={{ opacity: 0 }}
                        transition={{
                            delay: 1,
                            duration: 2.5,
                            ease: "easeInOut",
                        }}
                        src={`data:image/jpeg;base64,${tryOnOriginalResult}`}
                        alt="Original Result"
                        className="absolute inset-0 z-[100] h-full w-full object-contain"
                    />
                </div>
            </DialogContent>
        </Dialog>
    );
}
