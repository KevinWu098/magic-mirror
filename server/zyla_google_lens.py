import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("ZYLA_GOOGLE_LENS_API_KEY")

if not api_key:
    raise ValueError("API key not found. Make sure ZYLA_GOOGLE_LENS_API_KEY is set in .env file")

def search_google_lens(image_url):
    """
    Search for an image using Google Lens API via Zyla Labs
    
    Args:
        image_url (str): URL of the image to search
        
    Returns:
        dict: JSON response from the API
    """
    
    # API endpoint
    endpoint = "https://zylalabs.com/api/1338/google+lens+search+api/1119/search"
    
    # Request headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Request parameters
    params = {
        "url": image_url,
        "country": "us"
    }
    
    # Make the request
    response = requests.get(endpoint, headers=headers, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    # Dummy image URL for testing
    image_url = "https://i.postimg.cc/3dkZDMSS/sora-tank-shirt.png"
    
    # Call the function with dummy parameters
    result = search_google_lens(image_url)
    
    # Save the returned JSON to a file
    if result:
        with open("google_lens_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Results saved to google_lens_result.json")
    else:
        print("No results to save") 