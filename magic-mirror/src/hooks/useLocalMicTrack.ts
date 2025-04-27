import { useMemo } from "react";
import {
    TrackReferenceOrPlaceholder,
    useLocalParticipant,
} from "@livekit/components-react";
import { Track } from "livekit-client";

export default function useLocalMicTrack() {
    const { microphoneTrack, localParticipant } = useLocalParticipant();

    const micTrackRef: TrackReferenceOrPlaceholder = useMemo(() => {
        return {
            participant: localParticipant,
            source: Track.Source.Microphone,
            publication: microphoneTrack,
        };
    }, [localParticipant, microphoneTrack]);

    return micTrackRef;
}
