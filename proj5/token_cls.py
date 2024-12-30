# Token Classification
# https://huggingface.co/docs/transformers/tasks/token_classification#inference

# STEP 1
from transformers import pipeline

# STEP 2
classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")

# STEP 3
text = "The Golden State Warriors are an American professional basketball team based in San Francisco."

# STEP 4
result = classifier(text)

# STEP 5
print(result)


# 한국어 ver
classifier = pipeline("ner", model="Leo97/KoELECTRA-small-v3-modu-ner")
text = "내일 대전에 장미꽃을 사러 유재석과 함께 갈거야. 서울역으로 가는 방법 안내해줘"
result = classifier(text)
print(result)