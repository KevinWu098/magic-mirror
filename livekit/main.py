from typing import Any
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext
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
        super().__init__(instructions="You are a helpful voice AI assistant.")
        
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

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))