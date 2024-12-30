# Multiple Choice
# https://huggingface.co/docs/transformers/tasks/multiple_choice#inference

# STEP 1
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice
import torch

# STEP 2
model = AutoModelForMultipleChoice.from_pretrained("stevhliu/my_awesome_swag_model")
tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_swag_model")

# STEP 3
prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
candidate1 = "The law does not apply to croissants and brioche."
candidate2 = "The law applies to baguettes."

inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
labels = torch.tensor(0).unsqueeze(0)

# STEP 4
outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
logits = outputs.logits

# STEP 5
predicted_class = logits.argmax().item()
predicted_class
