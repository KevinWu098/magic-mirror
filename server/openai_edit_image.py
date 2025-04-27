import base64
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

prompt = "Make this shirt red"

start_time = time.time()

result = client.images.edit(
    model="gpt-image-1",
    image=[
        open("kev_globe_shirt.jpg", "rb")
    ],
    prompt=prompt,
    size="1024x1024",
    quality="medium",
)

end_time = time.time()

print(f"Edit API returned in {end_time - start_time} seconds")

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Save the image to a file
with open("edited_image.png", "wb") as f:
    f.write(image_bytes)