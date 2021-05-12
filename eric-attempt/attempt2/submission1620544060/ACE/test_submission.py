import json, sys

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').to(device)
model.classifier = nn.Sequential(
    nn.Linear(in_features=768, out_features=5, bias=True),
).to(device)
model.classifier.load_state_dict(torch.load('classifier.pt'))
model.eval()


def eval(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    logits = model(**inputs.to(device)).logits
    _, pred = torch.max(logits, 1)
    rating = float(pred.item()) + 1.0
    return rating

if len(sys.argv) > 1:
	validation_file = sys.argv[1]
	with open("output.jsonl", "w") as fw:
		with open(validation_file, "r") as fr:
			for line in fr:
				review = json.loads(line)
				fw.write(json.dumps({"review_id": review['review_id'], "predicted_stars": eval(review['text'])})+"\n")
	print("Output prediction file written")
else:
	print("No validation file given")
