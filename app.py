import os
import io
import json
import base64
import asyncio
import time
import numpy as np
from flask import Flask, request, Response
from flask_sock import Sock
from pydub import AudioSegment
import openai
from elevenlabs import ElevenLabs

# Load environment variables
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
TWILIO_APP_HOST = os.getenv("TWILIO_APP_HOST")

# Initialize clients
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Eller ersätt med din nyckel direkt
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

app = Flask(__name__)
sock = Sock(app)

SAMPLE_RATE = 8000
CHANNELS = 1
ELEVEN_VOICE = "bVMeCyTHy58xNoL34h3p"
ELEVEN_MODEL = "eleven_flash_v2"

SYSTEM_PROMPT = {"role": "system", "content": (
    "Du ringer ett cold call och ska sälja en höbal. "
    "Du ringer från Theos höbalar AB och du heter Mohammed. "
    "Sälj den för två hundra tusen kronor."
)}

conversations = {}

def ulaw_to_pcm(ulaw_bytes):
    ulaw = np.frombuffer(ulaw_bytes, dtype=np.uint8)
    ulaw = ulaw.astype(np.int16)

    BIAS = 0x84

    sign = ~ulaw & 0x80
    exponent = (ulaw >> 4) & 0x07
    mantissa = ulaw & 0x0F
    magnitude = ((mantissa << 4) + 0x08) << exponent
    pcm = (magnitude - BIAS) * np.where(sign == 0, 1, -1)

    return pcm.astype(np.int16).tobytes()

def pcm_to_ulaw(pcm_bytes):
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)

    BIAS = 0x84
    CLIP = 32635

    pcm = np.clip(pcm, -CLIP, CLIP)
    sign = (pcm >> 8) & 0x80
    pcm = np.where(sign != 0, -pcm, pcm)
    pcm = pcm + BIAS

    exponent = np.zeros_like(pcm)
    exp_lut = [0, 132, 396, 924, 1980, 4092, 8316, 16764]
    for i in range(7, 0, -1):
        exponent = np.where(pcm >= exp_lut[i], i, exponent)

    mantissa = (pcm >> (exponent + 3)) & 0x0F
    ulaw = ~(sign | (exponent << 4) | mantissa)

    return ulaw.astype(np.uint8).tobytes()

async def transcribe(wav_bytes):
    try:
        transcript = await client.audio.atranscribe("whisper-1", file=io.BytesIO(wav_bytes))
        print(f"[Whisper] Transcription: {transcript['text']}")
        return transcript["text"]
    except Exception as e:
        print("[Whisper Error]", e)
        return ""

async def chatgpt_reply(conv):
    try:
        response = await client.chat.completions.acreate(
            model="gpt-4o-mini",
            messages=conv,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        print(f"[GPT] Reply: {reply}")
        return reply
    except Exception as e:
        print("[GPT Error]", e)
        return "Jag kan inte svara just nu."

def tts_stream(text):
    print(f"[TTS] Synthesizing: {text}")
    return elevenlabs_client.generate(
        text=text,
        voice=ELEVEN_VOICE,
        model=ELEVEN_MODEL,
        stream=True,
    )

def generate_silence_chunk(duration_ms=250):
    """Generate a silent audio chunk in u-law encoded bytes for given duration."""
    silent_audio = AudioSegment.silent(duration=duration_ms, frame_rate=SAMPLE_RATE)
    raw_pcm = silent_audio.set_channels(CHANNELS).set_sample_width(2).raw_data
    mu = pcm_to_ulaw(raw_pcm)
    b64 = base64.b64encode(mu).decode()
    return b64

@app.route("/health")
def health():
    return "OK", 200

@app.route("/twiml", methods=["POST"])
def twiml():
    ws_url = f"wss://{TWILIO_APP_HOST}/ws"
    resp = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Response>\n'
        f'    <Start>\n'
        f'        <Stream url="{ws_url}" />\n'
        f'    </Start>\n'
        '    <Say>Du kopplas nu till vår AI-agent.</Say>\n'
        '</Response>'
    )
    return Response(resp, mimetype="text/xml")

@sock.route("/ws")
def ws(ws):
    call_sid = None
    buffer_pcm = bytearray()
    last_speech_time = time.time()
    silence_timeout = 4.5  # seconds
    fake_user_sent = False

    print("[WebSocket] Connected")

    while True:
        msg = ws.receive()
        now = time.time()

        if msg is None:
            print("[WebSocket] Disconnected")
            break

        packet = json.loads(msg)

        if not call_sid and packet.get("start"):
            call_sid = packet["start"]["callSid"]
            print(f"[Call Start] SID: {call_sid}")
            conversations[call_sid] = [SYSTEM_PROMPT]

            # Initial fake user input "hej"
            conversations[call_sid].append({"role": "user", "content": "hej"})
            reply = asyncio.run(chatgpt_reply(conversations[call_sid]))
            conversations[call_sid].append({"role": "assistant", "content": reply})

            # Send TTS greeting audio
            for chunk in tts_stream(reply):
                seg = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
                mu = pcm_to_ulaw(seg.raw_data)
                b64 = base64.b64encode(mu).decode()
                ws.send(json.dumps({"event": "media", "media": {"payload": b64}}))
            ws.send(json.dumps({"event": "mark", "name": "greeting_sent"}))
            last_speech_time = now
            fake_user_sent = False
            continue

        media = packet.get("media")
        if media:
            last_speech_time = now  # Reset silence timer
            raw = base64.b64decode(media["payload"])
            pcm = ulaw_to_pcm(raw)
            buffer_pcm.extend(pcm)

            # Process every 1 second of audio or more
            if len(buffer_pcm) >= SAMPLE_RATE * 2 * 1:
                seg = AudioSegment(
                    buffer_pcm,
                    frame_rate=SAMPLE_RATE,
                    sample_width=2,
                    channels=CHANNELS,
                )
                wav_io = io.BytesIO()
                seg.export(wav_io, format="wav")
                buffer_pcm.clear()

                text = asyncio.run(transcribe(wav_io.getvalue()))
                if text.strip():
                    conversations[call_sid].append({"role": "user", "content": text})
                    reply = asyncio.run(chatgpt_reply(conversations[call_sid]))
                    conversations[call_sid].append({"role": "assistant", "content": reply})

                    for chunk in tts_stream(reply):
                        seg2 = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                        seg2 = seg2.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
                        mu = pcm_to_ulaw(seg2.raw_data)
                        b64 = base64.b64encode(mu).decode()
                        ws.send(json.dumps({"event": "media", "media": {"payload": b64}}))

                fake_user_sent = False  # Reset flag if user actually spoke

        # Check silence timeout and if no fake user message was sent recently
        if now - last_speech_time > silence_timeout and not fake_user_sent and call_sid:
            print("[Silence] No user speech detected, sending fake input 'hej'")
            conversations[call_sid].append({"role": "user", "content": "hej"})
            reply = asyncio.run(chatgpt_reply(conversations[call_sid]))
            conversations[call_sid].append({"role": "assistant", "content": reply})

            for chunk in tts_stream(reply):
                seg2 = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                seg2 = seg2.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
                mu = pcm_to_ulaw(seg2.raw_data)
                b64 = base64.b64encode(mu).decode()
                ws.send(json.dumps({"event": "media", "media": {"payload": b64}}))

            last_speech_time = now
            fake_user_sent = True

        # Optional: send tiny silent chunks every second to keep connection alive
        # Could be added here if needed.

    print("[WebSocket] Connection closed")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
