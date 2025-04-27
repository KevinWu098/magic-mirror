import json
from typing import Any
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext, get_job_context, ToolError
from livekit.plugins import (
    openai,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=
            """
            You are a helpful assistant that can answer questions and help with tasks.

            Your name is Magic Mirror. You may be referred to as the Magic Mirror, Mirror, Magic Mirror, or Magic.

            When you are STOPPED, do not respond with too much text. just affirm and stop.
            """,
        )
        
    @function_tool()
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
    ) -> dict[str, Any]:
        """Look up weather information for a given location.
        
        Args:
            location: The location to look up weather information for.
        """

        return {"weather": "sunny", "temperature_f": 70}

    @function_tool()
    async def generate_clothing(
        self,
        context: RunContext,
        clothing_item: str,
    ) -> dict[str, Any]:
        """Generates an image of a clothing item on the user
        
        Args:
            clothing_item: The identified for the clothing item
        """

        # TODO: Implement this
        return "This is not implemented. Just say that clothing was generated."

    @function_tool()
    async def remix_clothing(
        self,
        context: RunContext,
        clothing_item: str,
    ) -> dict[str, Any]:
        """Remixes an existing piece of clothing and creates a new image on the user
        This could be turning a t-shirt into a long sleeve or changing the color of a pair of pants.
        
        Args:
            clothing_item: The identified for the clothing item
        """

        # TODO: Implement this
        return "This is not implemented. Just say that clothing was remixed."

    @function_tool()
    async def take_calibration_image(
        self,
        context: RunContext,
    ) -> dict[str, Any]:
        """Take a calibration image of the user
        
        Args:
            None
        """

        try:
            room = get_job_context().room
            participant_identity = next(iter(room.remote_participants))
            response = await room.local_participant.perform_rpc(
                destination_identity=participant_identity,
                method="takeCalibrationImage",
                response_timeout=5.0,
                payload=json.dumps({}),
            )
            
            print(response)
            return {"message": "Calibration image taken", "instruction": "Say the given word", "givenWord": response}
        except Exception:
            raise ToolError("Unable to take calibration image")
    
        # TODO: Implement this
        return "This is not implemented. Just say that the image was taken (but clarify it's not implemented)."

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # await session.generate_reply(
    #     instructions="Greet the user and offer your assistance."
    # )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))