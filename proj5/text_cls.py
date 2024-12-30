# Text Classification
# https://huggingface.co/docs/transformers/tasks/sequence_classification#inference

# STEP1: import modules
from transformers import pipeline

# STEP2: Create inference object
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")

# STEP3: Prepare data
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# STEP4: Inference
result = classifier(text)

# STEP5: Post-processing
print(result)   # {0: "NEGATIVE", 1: "POSITIVE"}

# 한국어 ver
# classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")
# result = classifier("현대바이오, '폴리탁셀' 코로나19 치료 가능성에 19% 급등")
# print(result)

# classifier = pipeline("sentiment-analysis", model="2tle/korean-curse-detection")
# result = classifier("식빵")
# print(result)