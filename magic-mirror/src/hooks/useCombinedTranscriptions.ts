import { useMemo } from "react";
import useLocalMicTrack from "@/hooks/useLocalMicTrack";
import {
    useTrackTranscription,
    useVoiceAssistant,
} from "@livekit/components-react";

export function useCombinedTranscriptions() {
    const { agentTranscriptions } = useVoiceAssistant();

    const micTrackRef = useLocalMicTrack();
    const { segments: userTranscriptions } = useTrackTranscription(micTrackRef);

    const combinedTranscriptions = useMemo(() => {
        return [
            ...agentTranscriptions.map((val) => {
                return { ...val, role: "assistant" };
            }),
            ...userTranscriptions.map((val) => {
                return { ...val, role: "user" };
            }),
        ].sort((a, b) => a.firstReceivedTime - b.firstReceivedTime);
    }, [agentTranscriptions, userTranscriptions]);

    return combinedTranscriptions;
}
