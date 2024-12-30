# STEP1: import modules
from typing import Annotated
from fastapi import FastAPI, Form
from transformers import pipeline
### https://pytorch.org/get-started/locally/
### pip install torch torchvision torchaudio

# STEP2: Create inference object
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")

app = FastAPI()

@app.post("/inference/")
async def inference(text: Annotated[str, Form()]):
    # STEP3: X
    # STEP4: Inference
    result = classifier(text)

    # STEP5: Post-processing
    print(result)
    return {"result": result}

# https://github.com/schibsted/WAAS
# https://sbert.net/