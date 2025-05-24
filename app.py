import os
import io
import json
import base64
import asyncio
import audioop
from flask import Flask, request, Response
from flask_sock import Sock
from pydub import AudioSegment
import openai
from elevenlabs import ElevenLabs

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
TWILIO_APP_HOST = os.getenv("TWILIO_APP_HOST")  # e.g., "your-app.onrender.com"

# Initialize clients
openai.api_key = OPENAI_API_KEY
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

# System prompt
SYSTEM_PROMPT = {"role": "system", "content": (
    "Du ringer ett cold call och ska sälja en höbal. "
    "Du ringer från Theos höbalar AB och du heter Mohammed. "
    "Sälj den för två hundra tusen kronor."
)}

# Conversation state per call
conversations = {}

@app.route("/health")
def health():
    return "OK", 200

@app.route("/twiml", methods=["POST"])
def twiml():
    ws_url = f"wss://{TWILIO_APP_HOST}/ws"
    print(f"[TWIML] Generated WebSocket URL: {ws_url}")
    resp = f""
<Response>
  <Start><Stream url="{ws_url}" /></Start>
  <Say>Du kopplas nu till vår AI-agent.</Say>
</Response>
""
    return Response(resp, mimetype="text/xml")

def ulaw_to_pcm(raw):
    return audioop.ulaw2lin(raw, 2)

def pcm_to_ulaw(pcm):
    return audioop.lin2ulaw(pcm, 2)

async def transcribe(wav_bytes):
    try:
        transcript = await openai.audio.atranscribe("whisper-1", file=io.BytesIO(wav_bytes))
        print(f"[Whisper] Transcription: {transcript['text']}")
        return transcript["text"]
    except Exception as e:
        print("[Whisper Error]", e)
        return ""

async def chatgpt_reply(conv):
    try:
        response = await openai.chat.completions.acreate(
            model="gpt-4o",
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
    print("[WebSocket] Connected")

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

            # Inject fake greeting
            conversations[call_sid].append({"role": "user", "content": "hej"})
            reply = asyncio.run(chatgpt_reply(conversations[call_sid]))
            conversations[call_sid].append({"role": "assistant", "content": reply})

            for chunk in tts_stream(reply):
                seg = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
                mu = pcm_to_ulaw(seg.raw_data)
                b64 = base64.b64encode(mu).decode()
                ws.send(json.dumps({"event": "media", "media": {"payload": b64}}))
            ws.send(json.dumps({"event": "mark", "name": "greeting_sent"}))
            continue

        media = packet.get("media")
        if media:
            raw = base64.b64decode(media["payload"])
            pcm = ulaw_to_pcm(raw)
            buffer_pcm.extend(pcm)

            if len(buffer_pcm) >= SAMPLE_RATE * 2 * 5:
                print("[Audio] Processing 5s chunk")
                seg = AudioSegment(
                    buffer_pcm,
                    frame_rate=SAMPLE_RATE,
                    sample_width=2,
                    channels=CHANNELS,
                )
                wav_io = io.BytesIO(); seg.export(wav_io, format="wav")
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

    print("[WebSocket] Session ended")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"[Server] Listening on port {port}")
    app.run(host="0.0.0.0", port=port)
