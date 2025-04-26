"use client";

import { useEffect, useState } from "react";
import { ConnectionDetails } from "@/app/api/connection-details/route";
import { onDeviceFailure } from "@/lib/livekit";
import { RoomAudioRenderer, RoomContext } from "@livekit/components-react";
import { Room, RoomEvent } from "livekit-client";

export function Client() {
    const [room] = useState(new Room());

    useEffect(() => {
        async function connect() {
            const url = new URL(
                process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT ??
                    "/api/connection-details",
                window.location.origin
            );
            const response = await fetch(url.toString());
            const connectionDetailsData: ConnectionDetails =
                await response.json();

            await room.connect(
                connectionDetailsData.serverUrl,
                connectionDetailsData.participantToken
            );
            await room.localParticipant.setMicrophoneEnabled(true);
        }

        // connect();

        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        room.on(RoomEvent.MediaDevicesError, onDeviceFailure);

        return () => {
            room.off(RoomEvent.MediaDevicesError, onDeviceFailure);
        };
    }, [room]);

    return (
        <div className="lk-room-container mx-auto">
            <RoomContext.Provider value={room}>
                <div>FOOBAR</div>
            </RoomContext.Provider>
        </div>
    );
}
