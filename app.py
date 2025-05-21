import os
import asyncio
import base64
import json
import io

from flask import Flask, request
from flask_sock import Sock
from pydub import AudioSegment
import audioop
import openai
from elevenlabs import ElevenLabs

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE = "bVMeCyTHy58xNoL34h3p"
ELEVENLABS_MODEL = "eleven_flash_v2"
SAMPLE_RATE = 8000
CHANNELS = 1

# Initialize Flask app and Flask-Sock
app = Flask(__name__)
sock = Sock(app)

# Initialize OpenAI and ElevenLabs
openai.api_key = OPENAI_API_KEY
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# System message for ChatGPT
SYSTEM_PROMPT = {
    "role": "system",
    "content": "Du ringer ett cold call och ska sälja en höbal. Du ringer från Theos höbalar AB och du heter Mohammed. Sälj den för två hundra tusen kronor."
}

conversations = {}

@app.route("/")
def index():
    return "Server is running."

@sock.route("/ws")
def ws_handler(ws):
    call_sid = None
    audio_buffer = bytearray()

    try:
        while True:
            message = ws.receive()
            if message is None:
                break

            data = json.loads(message)

            # Start event
            if not call_sid and "start" in data:
                call_sid = data["start"].get("callSid", "unknown")
                conversations[call_sid] = [SYSTEM_PROMPT]
                print(f"Call SID: {call_sid}")

            # Audio media
            if "media" in data:
                encoded_audio = data["media"]["payload"]
                raw_audio = base64.b64decode(encoded_audio)
                pcm_audio = audioop.ulaw2lin(raw_audio, 2)
                audio_buffer.extend(pcm_audio)

                if len(audio_buffer) >= SAMPLE_RATE * 2 * 5:
                    wav_audio = AudioSegment(
                        audio_buffer,
                        frame_rate=SAMPLE_RATE,
                        sample_width=2,
                        channels=CHANNELS
                    )
                    wav_io = io.BytesIO()
                    wav_audio.export(wav_io, format="wav")
                    wav_bytes = wav_io.getvalue()
                    audio_buffer.clear()

                    # Transcribe
                    transcript = transcribe_audio(wav_bytes)
                    print(f"User: {transcript}")
                    if transcript.strip():
                        conversations[call_sid].append({"role": "user", "content": transcript})

                        # Get response
                        reply = get_chatgpt_response(conversations[call_sid])
                        conversations[call_sid].append({"role": "assistant", "content": reply})
                        print(f"Bot: {reply}")

                        # TTS
                        for chunk in generate_tts_stream(reply):
                            segment = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                            segment = segment.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
                            pcm_bytes = segment.raw_data
                            mu_bytes = audioop.lin2ulaw(pcm_bytes, 2)
                            b64_audio = base64.b64encode(mu_bytes).decode("ascii")
                            ws.send(json.dumps({"event": "media", "media": {"payload": b64_audio}}))

                        ws.send(json.dumps({"event": "stop"}))
    except Exception as e:
        print(f"WebSocket error: {e}")


def transcribe_audio(wav_bytes):
    try:
        audio_file = io.BytesIO(wav_bytes)
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript["text"]
    except Exception as e:
        print("Whisper error:", e)
        return ""

def get_chatgpt_response(conversation):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=conversation,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print("ChatGPT error:", e)
        return "Jag kan inte svara just nu."

def generate_tts_stream(text):
    return elevenlabs_client.generate(
        text=text,
        voice=ELEVENLABS_VOICE,
        model=ELEVENLABS_MODEL,
        stream=True
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
