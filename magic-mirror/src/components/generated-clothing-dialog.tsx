import { Dispatch, SetStateAction } from "react";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Loader } from "lucide-react";

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
            <DialogContent className="max-w-none p-0 sm:max-w-[calc(100vw-32rem)]">
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
                        <img
                            src={`data:image/jpeg;base64,${generatedImage1}`}
                            alt="Generated clothing option"
                            className="h-full w-full object-cover"
                        />
                    ) : (
                        <div className="flex h-full items-center justify-center bg-gray-100">
                            <div className="flex flex-col items-center justify-center gap-2">
                                <div className="flex h-[1536px] w-[1024px] animate-pulse items-center justify-center rounded-lg border-4 border-gray-300 border-t-gray-600 bg-neutral-300">
                                    <Loader className="h-[15%] w-[15%] duration-1000" />
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </DialogContent>
        </Dialog>
    );
}
