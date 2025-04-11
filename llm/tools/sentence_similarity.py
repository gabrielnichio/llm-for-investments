import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name_or_path = "Alibaba-NLP/gte-multilingual-reranker-base"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, trust_remote_code=True,
    torch_dtype=torch.float16
)
model.eval()

pairs = [["中国的首都在哪儿","北京"], ["what is the capital of China?", "北京"], ["how to implement quick sort in python?","Introduction of quick sort"]]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)


