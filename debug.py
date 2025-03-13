import os
import requests
headers = {"xi-api-key": os.getenv("ELEVENLABS_API_KEY")}
voices_response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers)
print("Available voices:", voices_response.json())