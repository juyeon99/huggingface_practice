# Text Summarization
# https://huggingface.co/docs/transformers/tasks/summarization#inference

# STEP 1
from transformers import pipeline

# STEP 2
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")

# STEP 3
text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

# STEP 4
result = summarizer(text)

# STEP 5
print(text)
print(result)

# 한국어 ver
from transformers import pipeline
summarizer = pipeline("summarization", model="noahkim/KoT5_news_summarization")
text = "넷플릭스 시리즈 ‘오징어 게임’ 시즌2가 엇갈린 평에도 전 세계에서 압도적인 인기를 달리고 있다. 28일 글로벌 온라인동영상서비스(OTT) 순위 집계 회사 플릭스패트롤 누리집을 보면, 지난 26일 공개된 ‘오징어게임’ 시즌2는 27일 92개국에서 1위를 차지했다. 넷플릭스가 정식으로 서비스되는 나라 93곳 가운데, 뉴질랜드 1곳을 제외한 모든 나라에서 정상에 올랐다. 플릭스패트롤은 국가별로 기록한 순위를 점수로 환산해 총합을 집계한 총점도 매기고 있는데 현재 ‘오징어게임’ 시즌2는 929점을 받았다. 2위 ‘라 팔마’가 얻은 775점보다 높다."
result = summarizer(text)
print(result)