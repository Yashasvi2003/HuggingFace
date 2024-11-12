from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F


model_name="distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)

classifier=pipeline("sentiment-analysis",model=model_name, tokenizer=tokenizer)
results=classifier(["We are very happy to show you the happy face Transformers Library.",
                "We hope you don't hate them."])

for result in results:
 print(result)

 tokens=tokenizer.tokenize("We are happy to show you the happy face Transformers Library.")
 token_ids=tokenizer.convert_tokens_to_ids(tokens)
input_ids=tokenizer("We are happy to show you the happy face transformer library")


print(f'Token IDs: {tokens}')
print(f'Token IDs: {token_ids}')
print(f'Input IDs: {input_ids}')

X_train=["We are very happy to show you the happy face Transformers Library.",
                "We hope you don't hate them."]

batch=tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
 outputs=model(**batch,labels=torch.tensor([1,0]))
 print(outputs)
 predictions=torch.softmax(outputs.logits,dim=1)
 print(predictions)
 labels=torch.softmax(predictions.logits,dim=1)
 print(labels)
 labels=[model.config.id2label(label_id) for label_id in labels.toList()]
 print(labels)

# save_directory="saved"
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)

model_name="oliverghur/german-sentiment-bert"

# tokenizer.save_pretrained(save_directory)
# model=AutoModelForSequenceClassification.from_pretrained(save_directory)

tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForSequenceClassification.from_pretrained(model_name)

X_train_german=["Hugging Face ist eine großartige Plattform für die Verarbeitung natürlicher Sprache.Mit ihren Modellen und Tools können Entwickler schnell und einfach beeindruckende Anwendungen erstellen, die Texte verstehen und analysieren können. Es ist spannend zu sehen, wie diese Technologien uns helfen, Sprachbarrieren zu überwinden und Informationen besser zu verstehen."]

batch=tokenizer(X_train_german, padding=True, trunctaion=True, max_length=512, return_tensors="pt")
batch=torch.tensor(batch["input_ids"])
print(batch)

with torch.no_grad():
 ouputs=model(batch)
 label_ids=torch.argmax(ouputs, dim=1)
 print(label_ids)
 labels=[model.config.id2label[label_id] for label_id in labels.toList()]
 print(labels)