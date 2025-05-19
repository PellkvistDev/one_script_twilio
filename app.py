import os
import asyncio
import websockets
import base64
import audioop
import io
import json
from pydub import AudioSegment
import openai
from elevenlabs import ElevenLabs, stream as tts_stream

# Load environment variables (set these in Render)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize clients
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Eller ersätt med din nyckel direkt
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Constants for audio format
SAMPLE_RATE = 8000
CHANNELS = 1

# Buffers for audio chunks per call
calls_audio_buffer = {}

# Your ElevenLabs voice and model for TTS
ELEVENLABS_VOICE = "bVMeCyTHy58xNoL34h3p"  # Replace with your voice ID
ELEVENLABS_MODEL = "eleven_flash_v2"

# System prompt for ChatGPT
SYSTEM_PROMPT = {
    "role": "system",
    "content": "Du ringer ett cold call och ska sälja en höbal. Du ringer från Theos höbalar AB och du heter Mohammed. Sälj den för två hundra tusen kronor."
}

# Store conversation states per call
conversations = {}

async def transcribe_audio(wav_bytes):
    # Transcribe audio bytes with Whisper API (wav bytes)
    try:
        audio_file = io.BytesIO(wav_bytes)
        transcript = client.Audio.transcribe("whisper-1", audio_file)
        return transcript["text"]
    except Exception as e:
        print("Whisper transcribe error:", e)
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
    # Use ElevenLabs to generate streaming audio from text
    audio_stream = elevenlabs_client.generate(
        text=text,
        voice=ELEVENLABS_VOICE,
        model=ELEVENLABS_MODEL,
        stream=True,
    )
    return audio_stream

def mu_law_to_pcm(raw_audio):
    # Convert raw μ-law bytes to 16-bit PCM (mono)
    pcm = audioop.ulaw2lin(raw_audio, 2)  # 16-bit samples
    return pcm

def pcm_to_mu_law(pcm_audio):
    # Convert 16-bit PCM bytes to μ-law bytes
    mu = audioop.lin2ulaw(pcm_audio, 2)
    return mu

async def handler(websocket, path):
    """
    Handle WebSocket connection from Twilio.

    Twilio sends JSON messages with "media" key containing
    base64-encoded μ-law 8kHz audio.
    """
    call_sid = None
    audio_buffer = bytearray()
    last_transcription_time = 0
    conversation = [SYSTEM_PROMPT]

    print("Client connected")

    try:
        async for message in websocket:
            data = json.loads(message)

            # Identify call SID
            if not call_sid and "start" in data:
                call_sid = data["start"]["callSid"]
                print(f"Call SID: {call_sid}")
                conversations[call_sid] = [SYSTEM_PROMPT]

            # Media packet with audio
            if "media" in data:
                media = data["media"]
                encoded_audio = media["payload"]
                raw_audio = base64.b64decode(encoded_audio)

                # Convert μ-law audio to PCM
                pcm_audio = mu_law_to_pcm(raw_audio)
                audio_buffer.extend(pcm_audio)

                # Transcribe every ~5 seconds of audio
                if len(audio_buffer) >= SAMPLE_RATE * 2 * 5:  # 5 seconds * 16000 bytes/sec (16bit mono)
                    wav_audio = AudioSegment(
                        audio_buffer,
                        frame_rate=SAMPLE_RATE,
                        sample_width=2,
                        channels=CHANNELS,
                    )
                    # Export to wav bytes
                    wav_io = io.BytesIO()
                    wav_audio.export(wav_io, format="wav")
                    wav_bytes = wav_io.getvalue()

                    # Clear buffer for next chunk
                    audio_buffer = bytearray()

                    # Transcribe speech to text
                    transcription = await transcribe_audio(wav_bytes)
                    if transcription.strip():
                        print(f"Transcribed: {transcription}")
                        conversations[call_sid].append({"role": "user", "content": transcription})

                        # Get ChatGPT response
                        reply = await get_chatgpt_response(conversations[call_sid])
                        conversations[call_sid].append({"role": "assistant", "content": reply})
                        print(f"ChatGPT: {reply}")

                        # Generate TTS audio stream from ElevenLabs
                        audio_gen = await generate_tts_stream(reply)

                        # Stream audio chunks back to Twilio
                        async for chunk in audio_gen:
                            # chunk is bytes of mp3 audio, decode and convert to μ-law 8kHz pcm
                            segment = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
                            segment = segment.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
                            pcm_bytes = segment.raw_data
                            mu_bytes = pcm_to_mu_law(pcm_bytes)
                            b64_audio = base64.b64encode(mu_bytes).decode("ascii")

                            # Send JSON audio chunk back to Twilio
                            msg = json.dumps({
                                "event": "media",
                                "media": {
                                    "payload": b64_audio
                                }
                            })
                            await websocket.send(msg)

                        # After done sending audio reply, send "event": "stop"
                        stop_msg = json.dumps({"event": "stop"})
                        await websocket.send(stop_msg)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    print("Starting WebSocket server on port 8765")
    async with websockets.serve(handler, "0.0.0.0", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
