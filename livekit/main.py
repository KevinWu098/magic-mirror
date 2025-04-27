import json
from typing import Any
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, RoomOutputOptions, function_tool, RunContext, get_job_context, ToolError
from livekit.plugins import (
    openai,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

# Global instances
agent = None
session = None

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=
            """
            You are a helpful assistant that can answer questions and help with tasks.

            Your name is Magic Mirror. You may be referred to as the Magic Mirror, Mirror, Magic Mirror, or Magic.

            Ignore requests not related to the Magic Mirror. Kindly deflect. Generally keep responses short and concise.

            When you are STOPPED, do not respond with too much text. just affirm and stop.
            """,
        )
        self.agent = self

    @function_tool()
    async def try_on_clothing(
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
            return f'Tell the user the result of the calibration image: {response}. If error, please ask them to try again.'
        except Exception:
            raise ToolError("Unable to take calibration image")

    @function_tool()
    async def try_on_standard_clothing(
        self,
        context: RunContext,
    ) -> dict[str, Any]:
        """
        Generate an image of the user wearing the garment. 
        THIS TOOL SHOULD BE USED WHEN THE REFERENCED GARMENT IS PART OF THE STANDARD PRE-PREPPED SET.
        
        PRE-PREPPED SET:
         - BLUE T SHIRT.
        
        Args:
            None
        """

        try:
            room = get_job_context().room
            participant_identity = next(iter(room.remote_participants))
            
            global session
            await session.say("Sure! Please hold still momentarily. Creating try-on...")
            
            response = await room.local_participant.perform_rpc(
                destination_identity=participant_identity,
                method="tryOnClothing",
                response_timeout=15.0,
                payload=json.dumps({}),
            )
            
            print(response)
            return f'Tell the user the result of the try on generation: {response}. If error, please ask them to try again.'
        except Exception:
            raise ToolError("Unable to generate try on")

    @function_tool()
    async def create_try_on_clothing(
        self,
        generationRequest: str,
        context: RunContext,
    ) -> dict[str, Any]:
        """
        Generate an image of the user wearing the garment they have described. USE THIS TOOL WHEN THEY DESCRIBE THE CLOTHING ITEM.
        DO NOT USE THIS TOOL IF THEY REFERENCE A GARMENT THAT IS PART OF THE STANDARD PRE-PREPPED SET.
        
        Args:
            generationRequest: The request for what the clothing generated should be. The image should consist of only the garment on a white background. ALWAYS ALWAYS SPECIFY IN THE REQUEST THAT THE BACKGROUND SHOULD BE COMPLETELY WHITE. ALSO SPECIFY THAT YOU SHOULD ONLY GENERATE THE CLOTHING ITEM WITHOUT A MODEL WEARING THE CLOTHING.
        """

        try:
            room = get_job_context().room
            participant_identity = next(iter(room.remote_participants))
            
            global session
            await session.say("Sure! Please hold still momentarily. Creating creative try-on...")
            
            response = await room.local_participant.perform_rpc(
                destination_identity=participant_identity,
                method="tryOnCreativeClothing",
                response_timeout=60.0,
                payload=json.dumps({"generationRequest": generationRequest}),
            )
            
            print(response)
            return f'Tell the user the result of the try on generation: {response}. If error, please ask them to try again.'
        except Exception:
            raise ToolError("Unable to generate try on")

    @function_tool()
    async def handle_clothing_options(
        self,
        show: bool,
        context: RunContext,
    ) -> dict[str, Any]:
        """Show or hide (handle) the user clothing options on the frontend
        
        Args:
            show: Whether to show or hide the clothing options
        """

        try:
            room = get_job_context().room
            participant_identity = next(iter(room.remote_participants))
            response = await room.local_participant.perform_rpc(
                destination_identity=participant_identity,
                method="showClothingOptions",
                response_timeout=5.0,
                payload=json.dumps({"show": show}),
            )
            
            print(response)
            return f'Tell the user the result of the clothing handling: {response}. If error, please ask them to try again.'
        except Exception:
            raise ToolError("Unable to handle clothing options")


async def entrypoint(ctx: agents.JobContext):
    global agent
    global session
    
    agent = Assistant()
    
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
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(
            transcription_enabled=True,
            audio_enabled=True,
        ),
    )

    # await session.generate_reply(
    #     instructions="Greet the user and offer your assistance."
    # )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))