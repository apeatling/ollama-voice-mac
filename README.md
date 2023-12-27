# ollama-voice-mac
Mac compatible Ollama Voice building on the work of https://github.com/maudoin/ollama-voice

## Running

1. Install [Ollama](https://ollama.ai) on your Mac.
2. Download an [OpenAI Whisper Model](https://github.com/openai/whisper/discussions/63#discussioncomment-3798552) (base works fine).
3. Clone this repo somewhere.
4. Run `pip install -r requirements.txt` to install.
5. Run `python assistant.py` to start the assistant.

## Improving the voice

You can improve the quality of the voice by downloading a higher quality version. These instructions work on MacOS 14 Sonoma:

1. In System Settings select Accessibility > Spoken Content
2. Select System Voice and Manage Voices...
3. For English find "Zoe (Premium)" and download it.
4. Select Zoe (Premium) as your System voice.
