import os
import uuid

import gradio as gr
import whisper
from translate import Translator
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set")

client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)


def translator(audio_file):
    # 1. Transcribir Texto usando el repo de whisper de Open AI
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file, language="Spanish", fp16=False)
        transcription = result["text"]
    except Exception as e:
        raise gr.Error(f"se ha producido un error: {str(e)}")
    print(f"Texto original: {transcription}")

    # 2. Traducir texto con el traductor usando translate
    try:
        en_transcription = Translator(from_lang="es", to_lang="en").translate(transcription)
    except Exception as e:
        raise gr.Error(f"se ha producido un error traduciendo el texto: {str(e)}")
    print(f"Text traducido: {en_transcription}")

    # 3. Generar Audio traducido usando ElevenLabs
    try:
        response = client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=en_transcription,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        save_file_path = f"{uuid.uuid4()}.mp3"
        with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)
    except Exception as e:
        raise gr.Error(f"se ha producido un error creando el audio: {str(e)}")

    return save_file_path


web = gr.Interface(
    fn=translator,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Español"
    ),
    outputs=[gr.Audio(label="Ingles")],
    title="Traductor de voz",
    description="Traductor de voz con IA a varios lenguajes"
)

web.launch()
