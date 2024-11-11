from transformers import pipeline

import torch
import torch.nn.functional as F


model_name="distilbert-base-uncased-finetuned-sst-2-english"

classifier=pipeline("sentiment-analysis",model=model_name)
results=classifier(["We are very happy to show you the happy face Transformers Library.",
                "We hope you don't hate them."])

for result in results:
 print(result)