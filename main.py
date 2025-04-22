import torch
import os
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from transformers.utils import is_torch_sdpa_available
import time


print(is_torch_sdpa_available())
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v3.5"
test_filename = 'test_alana.ogg'

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa"
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,

)

data, samplerate = sf.read(test_filename)
test_data = {'array': data,"sampling_rate":samplerate}

output_file_path = os.path.splitext(test_filename)[0] + '.wav'

sf.write(output_file_path, data, samplerate)


dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

# result = pipe(sample)
now = time.time()
result_my_test = pipe(test_data)
print(time.time() - now)
print(result_my_test["text"])

