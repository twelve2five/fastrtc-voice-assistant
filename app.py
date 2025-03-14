import os
import time
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from fastrtc import (
    Stream,
    get_stt_model,
    ReplyOnPause,
    AdditionalOutputs
)

import requests
import io
import soundfile as sf
from gtts import gTTS
import re
import inspect
import torch
import torchaudio
import sys
from huggingface_hub import login, hf_hub_download

from deepseek import DeepSeekAPI

# Load environment variables
load_dotenv()

# Initialize clients
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
stt_model = get_stt_model()
deepseek_client = DeepSeekAPI(api_key=os.getenv("DEEPSEEK_API_KEY"))

# Add this debug code temporarily to see what methods are available:
print(dir(deepseek_client))

# Initialize CSM model
def initialize_csm():
    """Initialize the CSM text-to-speech model"""
    print("Initializing CSM model...")
    
    # Login to Hugging Face if token is available
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not found in environment variables")
        hf_token = input("Enter your Hugging Face token (needed for CSM model): ")
        login(token=hf_token)
    
    # Download and set up CSM
    try:
        # Add CSM to Python path if not already there
        if './csm' not in sys.path:
            sys.path.append('./csm')
        
        # Download the model checkpoint
        model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
        print(f"CSM model downloaded to: {model_path}")
        
        # Import and load the model
        from generator import load_csm_1b, Segment
        generator = load_csm_1b(model_path, "cuda" if torch.cuda.is_available() else "cpu")
        print("CSM model loaded successfully!")
        
        return generator
    except Exception as e:
        print(f"Error initializing CSM model: {e}")
        print("Falling back to gTTS...")
        return None

# Try to initialize CSM once at startup
csm_generator = initialize_csm()

def response(
    audio: tuple[int, np.ndarray],
    chatbot: list[dict] | None = None,
):
    chatbot = chatbot or []
    messages = [{"role": d["role"], "content": d["content"]} for d in chatbot]
    
    # Convert speech to text
    text = stt_model.stt(audio)
    print("prompt:", text)
    
    # Add user message to chat
    chatbot.append({"role": "user", "content": text})
    yield AdditionalOutputs(chatbot)
    
    # Get AI response
    messages.append({"role": "user", "content": text})
    response_text = get_deepseek_response(messages)
    
    # Add AI response to chat
    chatbot.append({"role": "assistant", "content": response_text})
    
    # Convert response to speech
    for audio_data in text_to_speech(response_text):
        if audio_data:
            yield audio_data
    
    yield AdditionalOutputs(chatbot)

# Create Gradio interface
chatbot = gr.Chatbot(type="messages")
stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(response, input_sample_rate=16000),
    additional_outputs_handler=lambda a, b: b,
    additional_inputs=[chatbot],
    additional_outputs=[chatbot],
    ui_args={"title": "LLM Voice Chat (Powered by DeepSeek & CSM)"}
)

# Create FastAPI app and mount stream
from fastapi import FastAPI
app = FastAPI()
app = gr.mount_gradio_app(app, stream.ui, path="/")
stream.mount(app)  # Mount the stream for telephone/fastphone integration

# Update the chat completion part based on available methods:
# We'll use direct HTTP requests as a fallback since the API structure is unclear:
def get_deepseek_response(messages):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512
    }
    response = requests.post(url, json=payload, headers=headers)
    
    # Check for error response
    if response.status_code != 200:
        print(f"DeepSeek API error: {response.status_code} - {response.text}")
        return "I'm sorry, I encountered an error processing your request."
        
    response_json = response.json()
    return response_json["choices"][0]["message"]["content"]

# Add CSM-based text_to_speech function
def text_to_speech(text):
    """Convert text to speech using CSM or gTTS as fallback"""
    try:
        # Split text into sentences for faster perceived response
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Use CSM if available
        if csm_generator is not None:
            print("Using CSM for text-to-speech...")
            
            # Process each sentence separately for faster perception
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                # Generate speech with CSM
                print(f"Generating CSM speech for: {sentence[:30]}...")
                audio = csm_generator.generate(
                    text=sentence,
                    speaker=0,  # Use speaker ID 0
                    context=[],
                    max_audio_length_ms=10_000,
                )
                
                # Convert to numpy array
                audio_np = audio.cpu().numpy()
                
                # Resample to 24000 Hz if needed (CSM uses 16000 Hz by default)
                if csm_generator.sample_rate != 24000:
                    audio_np = np.interp(
                        np.linspace(0, len(audio_np), int(len(audio_np) * 24000 / csm_generator.sample_rate)),
                        np.arange(len(audio_np)),
                        audio_np
                    )
                
                # Convert to 16-bit integers
                audio_np = (audio_np * 32767).astype(np.int16)
                
                # Ensure buffer size is even
                if len(audio_np) % 2 != 0:
                    audio_np = np.append(audio_np, [0])
                
                # Reshape and yield in chunks
                chunk_size = 4800
                for i in range(0, len(audio_np), chunk_size):
                    chunk = audio_np[i:i+chunk_size]
                    if len(chunk) > 0:
                        if len(chunk) % 2 != 0:
                            chunk = np.append(chunk, [0])
                        chunk = chunk.reshape(1, -1)
                        yield (24000, chunk)
                
        else:
            # Fall back to gTTS if CSM is not available
            print("Falling back to gTTS...")
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                # Process each sentence separately
                mp3_fp = io.BytesIO()
                
                # Force US English
                print(f"Using gTTS with en-us locale for sentence: {sentence[:20]}...")
                tts = gTTS(text=sentence, lang='en-us', tld='com', slow=False)
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                
                # Process audio data
                data, samplerate = sf.read(mp3_fp)
                
                # Convert to mono if stereo
                if len(data.shape) > 1 and data.shape[1] > 1:
                    data = data[:, 0]
                
                # Resample to 24000 Hz if needed
                if samplerate != 24000:
                    data = np.interp(
                        np.linspace(0, len(data), int(len(data) * 24000 / samplerate)),
                        np.arange(len(data)),
                        data
                    )
                
                # Convert to 16-bit integers
                data = (data * 32767).astype(np.int16)
                
                # Ensure buffer size is even
                if len(data) % 2 != 0:
                    data = np.append(data, [0])
                
                # Reshape and yield in chunks
                chunk_size = 4800
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i+chunk_size]
                    if len(chunk) > 0:
                        if len(chunk) % 2 != 0:
                            chunk = np.append(chunk, [0])
                        chunk = chunk.reshape(1, -1)
                        yield (24000, chunk)
    except Exception as e:
        print(f"Exception in text_to_speech: {e}")
        yield None

# Add this debug statement AFTER the function definition
print("text_to_speech function:", inspect.getsource(text_to_speech))

if __name__ == "__main__":
    os.environ["GRADIO_SSR_MODE"] = "false"
    
    # Check FastRTC version
    import fastrtc
    print(f"FastRTC version: {fastrtc.__version__ if hasattr(fastrtc, '__version__') else 'unknown'}")
    
    # Try running fastphone with additional diagnostic
    print("Starting phone service - attempting to inspect fastphone method...")
    import inspect
    print(f"FastPhone signature: {inspect.signature(stream.fastphone) if hasattr(stream, 'fastphone') else 'Not available'}")
    
    try:
        # Fix: Use keyword argument instead of positional
        phone_service = stream.fastphone(
            token=os.getenv("HF_TOKEN"),
            host="127.0.0.1",
            port=8000,
            share_server_tls_certificate=True  # Use keyword argument format
        )
        print("Phone service started successfully")
    except Exception as e:
        print(f"Error starting phone service: {e}")
        print("Falling back to web interface...")
        # Launch with web interface as fallback
        stream.ui.launch(server_port=7860)
