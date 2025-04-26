import { DeepgramContextProvider } from "@/context/deepgram-context";
import { MicrophoneContextProvider } from "@/context/microphone-context";

export default async function Layout({
    children,
}: Readonly<{ children: React.ReactNode }>) {
    return (
        <MicrophoneContextProvider>
            <DeepgramContextProvider>
                <div className="mx-auto flex aspect-[9/16] h-screen flex-col bg-white text-neutral-900">
                    {children}
                </div>
            </DeepgramContextProvider>
        </MicrophoneContextProvider>
    );
}
