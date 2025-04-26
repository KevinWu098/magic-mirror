import { useEffect, useRef, useState } from "react";
import {
    LiveConnectionState,
    LiveTranscriptionEvent,
    LiveTranscriptionEvents,
    useDeepgram,
} from "@/context/deepgram-context";
import {
    MicrophoneEvents,
    MicrophoneState,
    useMicrophone,
} from "@/context/microphone-context";
import { toast } from "sonner";

interface UseTranscriptionProps {
    onTranscript: (transcript: string, isFinal: boolean) => void;
}

export function useTranscription({ onTranscript }: UseTranscriptionProps) {
    const { connection, connectToDeepgram, connectionState, error } =
        useDeepgram();
    const {
        setupMicrophone,
        microphone,
        startMicrophone,
        microphoneState,
        stopMicrophone,
    } = useMicrophone();

    const [isListening, setIsListening] = useState(false);
    const unfinishedTextRef = useRef<string>("");
    const lastTranscriptTime = useRef<number>(Date.now());
    const pauseTimeout = useRef<NodeJS.Timeout>(undefined);
    const keepAliveInterval = useRef<NodeJS.Timeout>(undefined);

    useEffect(() => {
        setupMicrophone();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        if (microphoneState === MicrophoneState.Ready) {
            connectToDeepgram({
                model: "nova-2",
                interim_results: true,
                smart_format: true,
                endpointing: 2500,
                filler_words: true,
            });
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [microphoneState]);

    useEffect(() => {
        if (!microphone || !connection) return;

        const onData = (e: BlobEvent) => {
            if (e.data.size > 0 && connection) {
                connection.send(e.data);
            }
        };

        const onTranscriptData = (data: LiveTranscriptionEvent) => {
            const { is_final: isFinal } = data;
            const thisCaption = data.channel.alternatives.at(0)?.transcript;

            if (thisCaption) {
                lastTranscriptTime.current = Date.now();

                // Send partial transcription immediately
                if (!isFinal) {
                    onTranscript(thisCaption, false);
                }

                clearTimeout(pauseTimeout.current);
                pauseTimeout.current = setTimeout(() => {
                    if (Date.now() - lastTranscriptTime.current >= 1500) {
                        if (unfinishedTextRef.current) {
                            onTranscript(unfinishedTextRef.current, true);
                            unfinishedTextRef.current = "";
                        }
                    }
                }, 1500);

                if (isFinal) {
                    const updatedText = (
                        unfinishedTextRef.current +
                        " " +
                        thisCaption
                    ).trim();
                    unfinishedTextRef.current = updatedText;
                    onTranscript(updatedText, true);
                    unfinishedTextRef.current = "";
                }
            }
        };

        if (connectionState === LiveConnectionState.OPEN && isListening) {
            connection.addListener(
                LiveTranscriptionEvents.Transcript,
                onTranscriptData
            );
            microphone.addEventListener(MicrophoneEvents.DataAvailable, onData);
            if (microphone.state !== "recording") {
                startMicrophone();
            }
        }

        if (connectionState === LiveConnectionState.OPEN && !isListening) {
            stopMicrophone();
        }

        return () => {
            connection.removeListener(
                LiveTranscriptionEvents.Transcript,
                onTranscriptData
            );
            microphone.removeEventListener(
                MicrophoneEvents.DataAvailable,
                onData
            );
            clearTimeout(pauseTimeout.current);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [connectionState, isListening]);

    useEffect(() => {
        if (!connection) return;

        if (
            microphoneState !== MicrophoneState.Open &&
            connectionState === LiveConnectionState.OPEN
        ) {
            connection.keepAlive();
            keepAliveInterval.current = setInterval(() => {
                connection.keepAlive();
            }, 10000);
        } else {
            clearInterval(keepAliveInterval.current);
        }

        return () => clearInterval(keepAliveInterval.current);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [microphoneState, connectionState]);

    useEffect(() => {
        if (
            microphoneState === MicrophoneState.Open &&
            connectionState === LiveConnectionState.OPEN
        ) {
            setIsListening(true);
            toast.success("Microphone is now listening", {
                position: "top-center",
                richColors: true,
            });
        }
        if (
            microphoneState === MicrophoneState.Paused &&
            connectionState === LiveConnectionState.OPEN
        ) {
            setIsListening(false);
            toast.warning("Microphone is no longer listening", {
                position: "top-center",
                richColors: true,
            });
        }
    }, [microphoneState, connectionState]);

    useEffect(() => {
        if (error) {
            toast.error("Connection Error", {
                description: "Please try again later.",
                position: "top-center",
                richColors: true,
            });
        }
    }, [error]);

    const toggleListening = () => setIsListening(!isListening);

    return {
        isListening,
        toggleListening,
    };
}
