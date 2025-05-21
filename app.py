import os
import asyncio
import threading
import base64
import json
import io
import audioop
from flask import Flask
import openai
from elevenlabs import ElevenLabs, stream as tts_stream
from pydub import AudioSegment
import websockets

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# OpenAI and ElevenLabs clients
client = openai.OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Flask HTTP app for Render (port 10000)
app = Flask(__name__)

@app.route("/")
def home():
    return "OK", 200

# WebSocket audio settings
SAMPLE_RATE = 8000
CHANNELS = 1

# Per-call conversation states
conversations = {}

SYSTEM_PROMPT = {
    "role": "system",
    "content": "Du ringer ett cold call och ska sälja en höbal. Du ringer från Theos höbalar AB och du heter Mohammed. Sälj den för två hundra tusen kronor."
}

async def transcribe_audio(wav_bytes):
    try:
        audio_file = io.BytesIO(wav_bytes)
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return transcript.text
    except Exception as e:
        print("Whisper error:", e)
        return ""

async def get_chatgpt_response(conversation):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print("ChatGPT error:", e)
        return "Jag har problem med att svara just nu."

async def generate_tts_stream(text):
    audio_stream = elevenlabs_client.generate(
        text=text,
        voice="bVMeCyTHy58xNoL34h3p",
        model="eleven_flash_v2",
        stream=True,
    )
    return audio_stream

def mu_law_to_pcm(raw_audio):
    return audioop.ulaw2lin(raw_audio, 2)

def pcm_to_mu_law(pcm_audio):
    return audioop.lin2ulaw(pcm_audio, 2)

async def ws_handler(websocket):
    call_sid = None
    audio_buffer = bytearray()

    try:
        async for message in websocket:
            data = json.loads(message)

            if not call_sid and "start" in data:
                call_sid = data["start"]["callSid"]
                conversations[call_sid] = [SYSTEM_PROMPT]
                print(f"Call SID: {call_sid}")

            if "media" in data:
                media = data["media"]
                raw_audio = base64.b64decode(media["payload"])
                pcm_audio = mu_law_to_pcm(raw_audio)
                audio_buffer.extend(pcm_audio)

                if len(audio_buffer) >= SAMPLE_RATE * 2 * 5:  # 5 sec
                    segment = AudioSegment(
                        audio_buffer,
                        frame_rate=SAMPLE_RATE,
                        sample_width=2,
                        channels=1,
                    )
                    audio_io = io.BytesIO()
                    segment.export(audio_io, format="wav")
                    audio_buffer = bytearray()

                    transcription = await transcribe_audio(audio_io.getvalue())
                    if transcription.strip():
                        print("User:", transcription)
                        conversations[call_sid].append({"role": "user", "content": transcription})

                        reply = await get_chatgpt_response(conversations[call_sid])
                        conversations[call_sid].append({"role": "assistant", "content": reply})
                        print("Bot:", reply)

                        tts = await generate_tts_stream(reply)
                        async for chunk in tts:
                            segment = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                            segment = segment.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
                            mu = pcm_to_mu_law(segment.raw_data)
                            payload = base64.b64encode(mu).decode("ascii")
                            await websocket.send(json.dumps({"event": "media", "media": {"payload": payload}}))

                        await websocket.send(json.dumps({"event": "stop"}))
    except Exception as e:
        print("WebSocket error:", e)

# Launch WebSocket server on separate thread
def run_ws_server():
    async def server():
        print("WebSocket server running on port 8765")
        async with websockets.serve(ws_handler, "0.0.0.0", 8765):
            await asyncio.Future()

    asyncio.run(server())

if __name__ == "__main__":
    # Start WebSocket server in background
    threading.Thread(target=run_ws_server).start()
    # Start Flask HTTP server
    app.run(host="0.0.0.0", port=10000)
