import { Dispatch, SetStateAction } from "react";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Loader } from "lucide-react";
import { motion } from "motion/react";

export interface GeneratedClothingDialogProps {
    showModal: boolean;
    setShowModal: Dispatch<SetStateAction<boolean>>;
    generatedImage1: string | null;
    generatedImage2: string | null;
}

export function GeneratedClothingDialog({
    showModal,
    setShowModal,
    generatedImage1,
    generatedImage2,
}: GeneratedClothingDialogProps) {
    return (
        <Dialog
            open={showModal}
            onOpenChange={setShowModal}
        >
            <DialogContent className="max-w-none p-0 sm:max-w-[calc(100vw-16rem)]">
                {/* <DialogHeader>
                    <DialogTitle>Generated Clothing Options</DialogTitle>
                </DialogHeader> */}
                <div className="relative w-full overflow-hidden rounded-lg">
                    {generatedImage2 ? (
                        <img
                            src={`data:image/jpeg;base64,${generatedImage2}`}
                            alt="Generated clothing option"
                            className="h-full w-full object-cover"
                        />
                    ) : generatedImage1 ? (
                        <motion.img
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ duration: 1 }}
                            src={`data:image/jpeg;base64,${generatedImage1}`}
                            alt="Generated clothing option"
                            className="h-full w-full object-cover"
                        />
                    ) : null}
                </div>
            </DialogContent>
        </Dialog>
    );
}
