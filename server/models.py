import os
from google import genai

# Automatically picks up GEMINI_API_KEY or GOOGLE_API_KEY from environment variables
# Alternatively, pass it directly: genai.Client(api_key="YOUR_API_KEY")
client = genai.Client(api_key="AIzaSyA93tr6r64M6U6YDtZFgQf4PUspXhr-cQA")

print("Available Gemini Models:\n")
for m in client.models.list():
    print(f"Name: {m.name}")
    if hasattr(m, 'display_name') and m.display_name:
        print(f"Display Name: {m.display_name}")
    if hasattr(m, 'supported_actions') and m.supported_actions:
        print(f"Supported Actions: {m.supported_actions}")
    print("-" * 40)
