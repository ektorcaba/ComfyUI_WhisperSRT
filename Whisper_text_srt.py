import whisper
import os
import tempfile
import uuid
import folder_paths
import torchaudio
import numpy as np
import soundfile as sf  # pip install soundfile

class WhisperAudioToSRTText:
    languages_by_name = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),  # ComfyUI "File Input" node থেকে
                "model": (['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'],),
            },
            "optional": {
                "language": (
                    ["auto"] +
                    [s.capitalize() for s in sorted(list(whisper.tokenizer.LANGUAGES.values())) ],
                ),
                "prompt": ("STRING", {"default":""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("transcribed_text", "srt_text")
    FUNCTION = "transcribe"
    CATEGORY = "WhisperSRT"

    def transcribe(self, audio, model, language, prompt):

        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        audio_save_path = os.path.join(temp_dir, f"{uuid.uuid1()}.wav")
        torchaudio.save(audio_save_path, audio['waveform'].squeeze(
            0), audio["sample_rate"])
      
        print(f"[Whisper Node] Transcribing temporary .wav file: {audio_save_path}")
        print("[Whisper Node] Loading Whisper model...")
        w_model = whisper.load_model(model)

        transcribe_args = {"initial_prompt": prompt}

        if language != "auto":
            if WhisperAudioToSRTText.languages_by_name is None:
                WhisperAudioToSRTText.languages_by_name = {v.lower(): k for k, v in whisper.tokenizer.LANGUAGES.items()}
            transcribe_args['language'] = WhisperAudioToSRTText.languages_by_name[language.lower()]
        
        result = w_model.transcribe(audio_save_path, word_timestamps=True, **transcribe_args)
        
        segments = result["segments"]
        full_text = result["text"]
        srt_text = self.generate_srt_text(segments)
        
        os.remove(audio_save_path)

        return (full_text.strip(), srt_text.strip())

    def generate_srt_text(self, segments):
        def format_timestamp(seconds):
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

        srt_output = ""
        for idx, segment in enumerate(segments, 1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()

            srt_output += f"{idx}\n{start} --> {end}\n{text}\n\n"
        return srt_output
