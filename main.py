from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import os
import soundfile as sf
import io
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

app = FastAPI()

# Setup device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model and processor once when the server starts
model_id = "distil-whisper/distil-large-v3.5"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa"
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Read audio file into memory
    audio_bytes = await file.read()
    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # Prepare for ASR pipeline
    audio_input = {"array": audio_data, "sampling_rate": sample_rate}

    # Transcribe
    result = asr_pipeline(audio_input)
    return JSONResponse(content={"transcription": result["text"]})
