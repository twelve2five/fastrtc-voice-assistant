# FastRTC Voice Assistant

A real-time voice assistant using FastRTC for low-latency audio communication between browsers and a Python backend, powered by DeepSeek LLM and ElevenLabs voice synthesis.

## Overview

FastRTC Voice Assistant creates a seamless, responsive voice assistant experience by leveraging WebRTC technology for real-time audio streaming. The application connects browser clients to a Python backend server, where audio is processed through DeepSeek's large language model for understanding and generating responses, which are then converted to natural-sounding speech using ElevenLabs' voice synthesis technology.

## Features

- Real-time audio streaming using WebRTC for ultra-low latency
- Advanced speech recognition and natural language understanding with DeepSeek LLM
- High-quality, natural-sounding voice responses through ElevenLabs TTS
- Contextual conversation capabilities with memory of previous interactions
- Simple browser-based client interface
- Customizable voice personas and response styles

## Technical Stack

- **Frontend**: JavaScript, HTML/CSS
- **Backend**: Python with FastAPI
- **WebRTC**: For real-time audio communication
- **AI Models**:
  - DeepSeek LLM for natural language understanding and response generation
  - ElevenLabs for high-quality text-to-speech conversion

## Prerequisites

- Python 3.7+
- Modern web browser with WebRTC support (Chrome, Firefox, Edge, etc.)
- DeepSeek API key
- ElevenLabs API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/twelve2five/fastrtc-voice-assistant.git
   cd fastrtc-voice-assistant
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables for API keys:
   ```bash
   export DEEPSEEK_API_KEY="your_deepseek_api_key"
   export ELEVENLABS_API_KEY="your_elevenlabs_api_key"
   ```
   
   Alternatively, create a `.env` file in the project root directory:
   ```
   DEEPSEEK_API_KEY=your_deepseek_api_key
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   ```

## Configuration

### DeepSeek Configuration

The application uses DeepSeek's language model for natural language understanding and response generation. You can customize the model parameters in `config.py`:

- Model selection (DeepSeek-V2, etc.)
- Temperature and top-p settings for response generation
- System prompt and conversation context management
- Custom knowledge base integration

### ElevenLabs Configuration

Voice synthesis is handled by ElevenLabs' API. Configure voice settings in `config.py`:

- Voice ID selection
- Stability and similarity boost settings
- Speech rate and pitch adjustments
- Custom voice cloning options (if using premium features)

## Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to the displayed URL (typically http://localhost:8080)

3. Allow microphone access when prompted by your browser

4. Start speaking to interact with the voice assistant

5. The workflow is as follows:
   - Your voice is streamed in real-time to the server via WebRTC
   - Audio is transcribed and processed by DeepSeek's LLM
   - A response is generated based on your query and conversation history
   - ElevenLabs converts the text response to natural-sounding speech
   - The audio response is streamed back to your browser

## Advanced Usage

### Custom Voice Personas

Create different assistant personalities by editing the system prompts in `prompts.py`. You can define different conversation styles, knowledge domains, and personality traits.

### Extended Context

The assistant maintains conversation history to provide contextually relevant responses. You can adjust the context window size and management strategy in the configuration.

## Development

- `app.py` - Main application entry point
- `webrtc_handler.py` - Backend WebRTC connection handling
- `webrtc_client.js` - Frontend WebRTC implementation
- `llm_service.py` - DeepSeek LLM integration for text processing
- `tts_service.py` - ElevenLabs integration for speech synthesis
- `config.py` - Application configuration
- `debug.py` - Debugging utilities

## Debugging

For development and debugging purposes, you can run:
This provides additional logging and diagnostic information useful during development, including:
- WebRTC connection statistics
- DeepSeek API request/response logs
- ElevenLabs API request/response logs
- Audio processing metrics

## Performance Optimization

- Adjust audio sampling rates and buffer sizes in `config.py` to balance quality and latency
- Configure DeepSeek model parameters for faster response times
- Tune ElevenLabs voice settings for optimal audio quality

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgements

- [DeepSeek](https://deepseek.com) for their powerful language model
- [ElevenLabs](https://elevenlabs.io) for their natural-sounding text-to-speech technology
- FastRTC for enabling real-time communication capabilities
