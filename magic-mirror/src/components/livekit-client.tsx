"use client";

import { useCallback, useEffect, useState } from "react";
import { ConnectionDetails } from "@/app/api/connection-details/route";
import {
    BarVisualizer,
    DisconnectButton,
    RoomAudioRenderer,
    RoomContext,
    useVoiceAssistant,
    VideoTrack,
    VoiceAssistantControlBar,
} from "@livekit/components-react";
import { Room, RoomEvent } from "livekit-client";
import { XIcon } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

export function LivekitClient() {
    const [room] = useState(new Room());

    const onConnectButtonClicked = useCallback(async () => {
        // Generate room connection details, including:
        //   - A random Room name
        //   - A random Participant name
        //   - An Access Token to permit the participant to join the room
        //   - The URL of the LiveKit server to connect to
        //
        // In real-world application, you would likely allow the user to specify their
        // own participant name, and possibly to choose from existing rooms to join.

        const url = new URL(
            process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT ??
                "/api/connection-details",
            window.location.origin
        );
        const response = await fetch(url.toString());
        const connectionDetailsData: ConnectionDetails = await response.json();

        await room.connect(
            connectionDetailsData.serverUrl,
            connectionDetailsData.participantToken
        );
        await room.localParticipant.setMicrophoneEnabled(true);
    }, [room]);

    useEffect(() => {
        room.on(RoomEvent.MediaDevicesError, onDeviceFailure);

        return () => {
            room.off(RoomEvent.MediaDevicesError, onDeviceFailure);
        };
    }, [room]);

    return (
        <main
            data-lk-theme="default"
            className="grid h-full content-center bg-[var(--lk-bg)]"
        >
            <RoomContext.Provider value={room}>
                <div className="lk-room-container mx-auto max-h-[90vh] w-[90vw] max-w-[1024px]">
                    <SimpleVoiceAssistant
                        onConnectButtonClicked={onConnectButtonClicked}
                    />
                </div>
            </RoomContext.Provider>
        </main>
    );
}

function SimpleVoiceAssistant(props: { onConnectButtonClicked: () => void }) {
    const { state: agentState } = useVoiceAssistant();

    return (
        <>
            <AnimatePresence mode="wait">
                {agentState === "disconnected" ? (
                    <motion.div
                        key="disconnected"
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        transition={{
                            duration: 0.3,
                            ease: [0.09, 1.04, 0.245, 1.055],
                        }}
                        className="grid h-full items-center justify-center"
                    >
                        <motion.button
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ duration: 0.3, delay: 0.1 }}
                            className="rounded-md bg-white px-4 py-2 text-black uppercase"
                            onClick={() => props.onConnectButtonClicked()}
                        >
                            Start a conversation
                        </motion.button>
                    </motion.div>
                ) : (
                    <motion.div
                        key="connected"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        transition={{
                            duration: 0.3,
                            ease: [0.09, 1.04, 0.245, 1.055],
                        }}
                        className="flex h-full flex-col items-center gap-4"
                    >
                        <AgentVisualizer />
                        <div className="w-full flex-1">
                            {/* <TranscriptionView /> */}
                        </div>
                        <div className="w-full">
                            <ControlBar
                                onConnectButtonClicked={
                                    props.onConnectButtonClicked
                                }
                            />
                        </div>
                        <RoomAudioRenderer />
                        {/* <NoAgentNotification state={agentState} /> */}
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
}

function AgentVisualizer() {
    const { state: agentState, videoTrack, audioTrack } = useVoiceAssistant();

    if (videoTrack) {
        return (
            <div className="h-[512px] w-[512px] overflow-hidden rounded-lg">
                <VideoTrack trackRef={videoTrack} />
            </div>
        );
    }
    return (
        <div className="h-[300px] w-full">
            <BarVisualizer
                state={agentState}
                barCount={5}
                trackRef={audioTrack}
                className="agent-visualizer"
                options={{ minHeight: 24 }}
            />
        </div>
    );
}

function ControlBar(props: { onConnectButtonClicked: () => void }) {
    const { state: agentState } = useVoiceAssistant();

    return (
        <div className="relative h-[60px]">
            <AnimatePresence>
                {agentState === "disconnected" && (
                    <motion.button
                        initial={{ opacity: 0, top: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0, top: "-10px" }}
                        transition={{
                            duration: 1,
                            ease: [0.09, 1.04, 0.245, 1.055],
                        }}
                        className="absolute left-1/2 -translate-x-1/2 rounded-md bg-white px-4 py-2 text-black uppercase"
                        onClick={() => props.onConnectButtonClicked()}
                    >
                        Start a conversation
                    </motion.button>
                )}
            </AnimatePresence>
            <AnimatePresence>
                {agentState !== "disconnected" &&
                    agentState !== "connecting" && (
                        <motion.div
                            initial={{ opacity: 0, top: "10px" }}
                            animate={{ opacity: 1, top: 0 }}
                            exit={{ opacity: 0, top: "-10px" }}
                            transition={{
                                duration: 0.4,
                                ease: [0.09, 1.04, 0.245, 1.055],
                            }}
                            className="absolute left-1/2 flex h-8 -translate-x-1/2 justify-center"
                        >
                            <VoiceAssistantControlBar
                                controls={{ leave: false }}
                            />
                            <DisconnectButton>
                                <XIcon />
                            </DisconnectButton>
                        </motion.div>
                    )}
            </AnimatePresence>
        </div>
    );
}

function onDeviceFailure(error: Error) {
    console.error(error);
    alert(
        "Error acquiring camera or microphone permissions. Please make sure you grant the necessary permissions in your browser and reload the tab"
    );
}
