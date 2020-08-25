import os
import sys
sys.path.insert(0, os.path.abspath('../'))

from transformers import pipeline, AutoModelForSequenceClassification
from tokenization_kbalbert import KbAlbertCharTokenizer

kb_albert_model_path = '../kb-albert-model'
model_output_path = 'nsmc_outputs'
model = AutoModelForSequenceClassification.from_pretrained(model_output_path)
tokenizer = KbAlbertCharTokenizer.from_pretrained(kb_albert_model_path)

nsmc_classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, framework='pt')

reviews = ['이 영화 최악이었어!',
           '볼거리가 많은 내 인생 영화 ㅎㅎ']
results = nsmc_classifier(reviews)
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")