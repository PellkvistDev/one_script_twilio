```python
import os
import io
import json
import base64
import audioop
import asyncio
from flask import Flask, request, Response
from flask_sock import Sock
from pydub import AudioSegment
import openai
from elevenlabs import ElevenLabs

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
TWILIO_APP_HOST = os.getenv("TWILIO_APP_HOST")  # e.g. "your-app.onrender.com"

# Initialize clients
openai.api_key = OPENAI_API_KEY
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Flask + WebSocket setup
app = Flask(__name__)
sock = Sock(app)

# Audio settings
SAMPLE_RATE = 8000
CHANNELS = 1

# ElevenLabs TTS config
ELEVEN_VOICE = "bVMeCyTHy58xNoL34h3p"
ELEVEN_MODEL = "eleven_flash_v2"

# System prompt for ChatGPT
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Du ringer ett cold call och ska sälja en höbal. "
        "Du ringer från Theos höbalar AB och du heter Mohammed. "
        "Sälj den för två hundra tusen kronor."
    )
}

# Store conversations per call
conversations = {}

# Health check endpoint
@app.route("/health")
def health():
    return "OK", 200

# TwiML endpoint for Twilio to start the stream
@app.route("/twiml", methods=["POST"])
def twiml():
    ws_url = f"wss://{TWILIO_APP_HOST}/ws"
    twiml_response = (
        "<Response>"
        f"<Start><Stream url=\"{ws_url}\"/></Start>"
        "<Say>Du kopplas nu till vår AI-agent.</Say>"
        "</Response>"
    )
    return Response(twiml_response, mimetype="text/xml")

# Audio conversion helpers

def ulaw_to_pcm(raw_audio: bytes) -> bytes:
    return audioop.ulaw2lin(raw_audio, 2)

def pcm_to_ulaw(pcm_audio: bytes) -> bytes:
    return audioop.lin2ulaw(pcm_audio, 2)

# Transcribe audio using Whisper
async def transcribe_audio(wav_bytes: bytes) -> str:
    try:
        transcript = openai.Audio.transcribe("whisper-1", io.BytesIO(wav_bytes))
        return transcript["text"]
    except Exception as e:
        print("Whisper error:", e)
        return ""

# Get ChatGPT response
async def get_chatgpt_response(conv: list) -> str:
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=conv,
            temperature=0.7
        )
        return res["choices"][0]["message"]["content"]
    except Exception as e:
        print("ChatGPT error:", e)
        return "Jag kan inte svara just nu."

# Generate streamed TTS audio
def tts_stream(text: str):
    return elevenlabs_client.generate(
        text=text,
        voice=ELEVEN_VOICE,
        model=ELEVEN_MODEL,
        stream=True
    )

# WebSocket handler for Twilio <Stream>
@sock.route("/ws")
def websocket_handler(ws):
    call_sid = None
    buffer_pcm = bytearray()

    while True:
        msg = ws.receive()
        if msg is None:
            break

        packet = json.loads(msg)

        # Handle start event
        if not call_sid and packet.get("start"):
            call_sid = packet["start"]["callSid"]
            conversations[call_sid] = [SYSTEM_PROMPT]
            print(f"Call SID: {call_sid}")
            continue

        # Handle media packets
        media = packet.get("media")
        if media:
            raw = base64.b64decode(media["payload"])
            pcm = ulaw_to_pcm(raw)
            buffer_pcm.extend(pcm)

            # Process every ~5 seconds of audio
            bytes_per_second = SAMPLE_RATE * CHANNELS * 2
            if len(buffer_pcm) >= bytes_per_second * 5:
                # Convert PCM buffer to WAV bytes
                segment = AudioSegment(
                    buffer_pcm,
                    frame_rate=SAMPLE_RATE,
                    sample_width=2,
                    channels=CHANNELS
                )
                wav_io = io.BytesIO()
                segment.export(wav_io, format="wav")
                wav_bytes = wav_io.getvalue()
                buffer_pcm.clear()

                # Transcribe
                transcription = asyncio.run(transcribe_audio(wav_bytes))
                print("Transcribed:", transcription)
                if transcription.strip():
                    conversations[call_sid].append({"role": "user", "content": transcription})

                    # ChatGPT reply
                    reply = asyncio.run(get_chatgpt_response(conversations[call_sid]))
                    conversations[call_sid].append({"role": "assistant", "content": reply})
                    print("GPT reply:", reply)

                    # TTS and send back
                    for chunk in tts_stream(reply):
                        seg = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                        seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
                        mu_law = pcm_to_ulaw(seg.raw_data)
                        b64 = base64.b64encode(mu_law).decode("ascii")
                        ws.send(json.dumps({"event": "media", "media": {"payload": b64}}))

                    # Signal end of message
                    ws.send(json.dumps({"event": "stop"}))

    print("WebSocket connection closed")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
```
