import os
import io
import json
import base64
import asyncio
import audioop
from flask import Flask, request, Response
from pydub import AudioSegment
import websockets
import threading

from openai import OpenAI
from elevenlabs import ElevenLabs, stream as tts_stream

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Flask app for Twilio webhook
app = Flask(__name__)

# WebSocket settings
SAMPLE_RATE = 8000
CHANNELS = 1
ELEVENLABS_VOICE = "bVMeCyTHy58xNoL34h3p"
ELEVENLABS_MODEL = "eleven_flash_v2"
SYSTEM_PROMPT = {
    "role": "system",
    "content": "Du ringer ett cold call och ska sälja en höbal. Du ringer från Theos höbalar AB och du heter Mohammed. Sälj den för två hundra tusen kronor."
}
conversations = {}

# TwiML route (used in Twilio call)
@app.route("/twiml", methods=["POST"])
def twiml():
    stream_url = "wss://twiliocalls-hvq2.onrender.com/ws"
    response = f"""
    <Response>
        <Start>
            <Stream url="{stream_url}" />
        </Start>
        <Say>Du kopplas nu till vår AI-agent.</Say>
    </Response>
    """
    return Response(response, mimetype="text/xml")

# Audio conversion utils
def mu_law_to_pcm(raw_audio):
    return audioop.ulaw2lin(raw_audio, 2)

def pcm_to_mu_law(pcm_audio):
    return audioop.lin2ulaw(pcm_audio, 2)

# Whisper transcription
async def transcribe_audio(wav_bytes):
    try:
        audio_file = io.BytesIO(wav_bytes)
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript.text
    except Exception as e:
        print("Whisper error:", e)
        return ""

# ChatGPT reply
async def get_chatgpt_response(convo):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=convo,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print("ChatGPT error:", e)
        return "Jag har problem med att svara just nu."

# ElevenLabs TTS
async def generate_tts_stream(text):
    return elevenlabs_client.generate(
        text=text,
        voice=ELEVENLABS_VOICE,
        model=ELEVENLABS_MODEL,
        stream=True
    )

# WebSocket handler
async def ws_handler(websocket):
    call_sid = None
    audio_buffer = bytearray()
    print("WebSocket client connected")

    try:
        async for message in websocket:
            data = json.loads(message)

            if "start" in data:
                call_sid = data["start"]["callSid"]
                conversations[call_sid] = [SYSTEM_PROMPT]
                continue

            if "media" in data:
                payload = data["media"]["payload"]
                raw_audio = base64.b64decode(payload)
                pcm_audio = mu_law_to_pcm(raw_audio)
                audio_buffer.extend(pcm_audio)

                if len(audio_buffer) >= SAMPLE_RATE * 2 * 5:
                    segment = AudioSegment(
                        audio_buffer,
                        frame_rate=SAMPLE_RATE,
                        sample_width=2,
                        channels=CHANNELS
                    )
                    audio_io = io.BytesIO()
                    segment.export(audio_io, format="wav")
                    wav_bytes = audio_io.getvalue()
                    audio_buffer.clear()

                    text = await transcribe_audio(wav_bytes)
                    if text.strip():
                        print("User said:", text)
                        conversations[call_sid].append({"role": "user", "content": text})

                        reply = await get_chatgpt_response(conversations[call_sid])
                        print("GPT:", reply)
                        conversations[call_sid].append({"role": "assistant", "content": reply})

                        audio_stream = await generate_tts_stream(reply)
                        async for chunk in audio_stream:
                            segment = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                            segment = segment.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
                            pcm_bytes = segment.raw_data
                            mu_bytes = pcm_to_mu_law(pcm_bytes)
                            b64_audio = base64.b64encode(mu_bytes).decode("ascii")

                            response_msg = json.dumps({
                                "event": "media",
                                "media": {"payload": b64_audio}
                            })
                            await websocket.send(response_msg)

                        await websocket.send(json.dumps({"event": "stop"}))

    except Exception as e:
        print("WebSocket error:", e)

# Start WebSocket server in thread
def start_websocket_server():
    async def server():
        print("WebSocket server running on port 8765")
        async with websockets.serve(ws_handler, "0.0.0.0", 8765, process_request=reject_head):
            await asyncio.Future()
    asyncio.run(server())

# Workaround to reject HEAD requests (Render/health check)
async def reject_head(path, request_headers):
    if request_headers.get("method", "") == "HEAD":
        raise websockets.exceptions.InvalidMessage("HEAD not allowed")

# Main entry
if __name__ == "__main__":
    threading.Thread(target=start_websocket_server).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
