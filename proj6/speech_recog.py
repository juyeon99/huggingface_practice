# Speech Recognition
# https://huggingface.co/openai/whisper-large-v3-turbo

# STEP 1
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

# STEP 2
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True, # 30초보다 긴 음성 처리 가능
)
# ==> pipe = pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo")

# STEP 3
dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

# STEP 4
result = pipe(sample)

# STEP 5
print(result["text"])


# 한국어 ver
# !wget https://download.blog.naver.com/open/0e9b12a1bbefea361bff98a8947f0e74d485789e84/61_5JfleKw-hM29Bjwoggl2-gwivsdxpL0IZ6pjuOcdiv3wCYFMDJj-kzT5lkPMhSaIyVJ7xFIE0HjUvKvCOSQ/광화문자생한방병원_1.mp3
# result = pipe("광화문자생한방병원_1.mp3")
# print(result["text"])