import os
import io
import json
import base64
import asyncio
import numpy as np
from flask import Flask, request, Response
from flask_sock import Sock
from pydub import AudioSegment
from openai import OpenAI
from elevenlabs import ElevenLabs
from twilio.twiml.voice_response import Connect, VoiceResponse, Say, Stream

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
TWILIO_APP_HOST = os.getenv("TWILIO_APP_HOST")  # e.g., "your-app.onrender.com"

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Flask app and WebSocket
app = Flask(__name__)
sock = Sock(app)

# Audio settings
SAMPLE_RATE = 8000
CHANNELS = 1

# ElevenLabs TTS settings
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

# Conversation state per call
conversations = {}

# Silence timeout in seconds before sending fake 'hej'
SILENCE_TIMEOUT = 4.5

@app.route("/health")
def health():
    return "OK", 200

@app.route("/twiml", methods=["POST"])
def twiml():
    ws_url = f"wss://one-script-twilio.onrender.com/ws"
    resp = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Response>\n'
        '    <Connect>\n'
        f'        <Stream url="{ws_url}" />\n'
        '    </Connect>\n'
        '    <Say>Du kopplas nu till vår AI-agent.</Say>\n'
        '</Response>'
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

@sock.route("/ws")
def ws(ws):
    call_sid = None
    buffer_pcm = bytearray()
    last_audio_time = asyncio.get_event_loop().time()

    print("[WebSocket] Connected")

    async def send_fake_hej():
        nonlocal last_audio_time
        if call_sid is None:
            return
        now = asyncio.get_event_loop().time()
        if now - last_audio_time >= SILENCE_TIMEOUT:
            print("[Silence] No audio received, sending fake 'hej'")
            conversations[call_sid].append({"role": "user", "content": "hej"})
            reply = await chatgpt_reply(conversations[call_sid])
            conversations[call_sid].append({"role": "assistant", "content": reply})

            for chunk in tts_stream(reply):
                seg = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
                mu = pcm_to_ulaw(seg.raw_data)
                b64 = base64.b64encode(mu).decode()
                ws.send(json.dumps({"event": "media", "media": {"payload": b64}}))

            last_audio_time = now

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                print("[WebSocket] Disconnected")
                break

            packet = json.loads(msg)

            if not call_sid and packet.get("start"):
                call_sid = packet["start"]["callSid"]
                print(f"[Call Start] SID: {call_sid}")
                conversations[call_sid] = [SYSTEM_PROMPT]

                # Inject fake greeting immediately so call won't hang up
                conversations[call_sid].append({"role": "user", "content": "hej"})
                reply = loop.run_until_complete(chatgpt_reply(conversations[call_sid]))
                conversations[call_sid].append({"role": "assistant", "content": reply})

                for chunk in tts_stream(reply):
                    seg = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                    seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
                    mu = pcm_to_ulaw(seg.raw_data)
                    b64 = base64.b64encode(mu).decode()
                    ws.send(json.dumps({"event": "media", "media": {"payload": b64}}))
                ws.send(json.dumps({"event": "mark", "name": "greeting_sent"}))
                last_audio_time = loop.time()
                continue

            media = packet.get("media")
            if media:
                raw = base64.b64decode(media["payload"])
                pcm = ulaw_to_pcm(raw)
                buffer_pcm.extend(pcm)
                last_audio_time = loop.time()

                if len(buffer_pcm) >= SAMPLE_RATE * 2 * 5:  # 5 seconds of audio
                    print("[Audio] Processing 5s chunk")
                    seg = AudioSegment(
                        buffer_pcm,
                        frame_rate=SAMPLE_RATE,
                        sample_width=2,
                        channels=CHANNELS,
                    )
                    wav_io = io.BytesIO()
                    seg.export(wav_io, format="wav")
                    buffer_pcm.clear()

                    text = loop.run_until_complete(transcribe(wav_io.getvalue()))
                    if text.strip():
                        conversations[call_sid].append({"role": "user", "content": text})
                        reply = loop.run_until_complete(chatgpt_reply(conversations[call_sid]))
                        conversations[call_sid].append({"role": "assistant", "content": reply})

                        for chunk in tts_stream(reply):
                            seg2 = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                            seg2 = seg2.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
                            mu = pcm_to_ulaw(seg2.raw_data)
                            b64 = base64.b64encode(mu).decode()
                            ws.send(json.dumps({"event": "media", "media": {"payload": b64}}))

            else:
                # Check silence timeout periodically
                loop.run_until_complete(send_fake_hej())

    except Exception as e:
        print("[WebSocket] Error:", e)

    print("[WebSocket] Session ended")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"[Server] Listening on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
