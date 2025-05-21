import os
import io
import json
import base64
import asyncio
import numpy as np
from flask import Flask, request, Response
from flask_sock import Sock
from pydub import AudioSegment
import openai
from elevenlabs import ElevenLabs

# Initialize OpenAI and ElevenLabs clients
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Eller ersätt med din nyckel direkt
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Flask app setup
app = Flask(__name__)
sock = Sock(app)

# Constants
SAMPLE_RATE = 8000
CHANNELS = 1
ELEVEN_VOICE = "bVMeCyTHy58xNoL34h3p"
ELEVEN_MODEL = "eleven_flash_v2"
TWILIO_APP_HOST = os.getenv("TWILIO_APP_HOST")

# System prompt for the AI
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Du ringer ett cold call och ska sälja en höbal. "
        "Du ringer från Theos höbalar AB och du heter Mohammed. "
        "Sälj den för två hundra tusen kronor."
    )
}

# Dictionary to hold conversation histories
conversations = {}

@app.route("/health")
def health():
    return "OK", 200

@app.route("/twiml", methods=["POST"])
def twiml():
    ws_url = f"wss://{TWILIO_APP_HOST}/ws"
    resp = (
        f"<Response>"
        f"<Start><Stream url=\"{ws_url}\"/></Start>"
        f"<Say>Du kopplas nu till vår AI-agent.</Say>"
        f"</Response>"
    )
    return Response(resp, mimetype="text/xml")

def ulaw_to_pcm(ulaw_bytes):
    ulaw = np.frombuffer(ulaw_bytes, dtype=np.uint8)
    ulaw = ulaw.astype(np.int16)

    BIAS = 0x84
    MULAW_MAX = 0x1FFF

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
        transcript = client.Audio.transcribe("whisper-1", io.BytesIO(wav_bytes))
        return transcript["text"]
    except Exception as e:
        print("Whisper error:", e)
        return ""

async def chatgpt_reply(conv):
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conv,
            temperature=0.7,
        )
        return res.choices[0].message.content
    except Exception as e:
        print("GPT error:", e)
        return "Jag kan inte svara just nu."

def tts_stream(text):
    return elevenlabs_client.generate(
        text=text,
        voice=ELEVEN_VOICE,
        model=ELEVEN_MODEL,
        stream=True,
    )

@sock.route("/ws")
def ws(ws):
    call_sid = None
    buffer_pcm = bytearray()
    while True:
        msg = ws.receive()
        if msg is None:
            break
        packet = json.loads(msg)

        if not call_sid and packet.get("start"):
            call_sid = packet["start"]["callSid"]
            conversations[call_sid] = [SYSTEM_PROMPT]

            # Simulate "hej" as first message
            conversations[call_sid].append({"role": "user", "content": "hej"})
            reply = asyncio.run(chatgpt_reply(conversations[call_sid]))
            conversations[call_sid].append({"role": "assistant", "content": reply})

            # Send TTS reply
            for chunk in tts_stream(reply):
                seg2 = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                seg2 = seg2.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
                mu = pcm_to_ulaw(seg2.raw_data)
                b64 = base64.b64encode(mu).decode()
                ws.send(json.dumps({"event": "media", "media": {"payload": b64}}))
            continue

        media = packet.get("media")
        if media:
            raw = base64.b64decode(media["payload"])
            pcm = ulaw_to_pcm(raw)
            buffer_pcm.extend(pcm)

            if len(buffer_pcm) >= SAMPLE_RATE * 2 * 5:
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

                    ws.send(json.dumps({"event": "stop"}))
    print("WebSocket closed")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
