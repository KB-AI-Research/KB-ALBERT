# KB-ALBERT-SUBTOKEN
SubToken-based KB-ALBERT Model and Tokenizer 

## 모델 상세 정보

### 1. Architecture

- max_seq_length=512
- embedding_size=128
- hidden_size=768
- num_hidden_layers = 12
- vocab_size = 50000

### 2. 학습 데이터 셋

- 일반 도메인 텍스트(위키 + 뉴스 등) : 약 25GB 
- 금융 도메인 텍스트(경제/금융 특화 뉴스 + 리포트 등) : 약 15GB

</br>


## How-to-Use

### 1. Model Download

- KB-ALBERT를 사용하시고자 하시는 분들은 아래 메일로 소속, 이름, 사용용도를 간단히 작성하셔서 발송해 주세요.
- ai.kbg@kbfg.com


### 2. Source Download and Install

```shell script
git clone
cd kb-albert-subtoken
pip install -U python-crfsuite, transformers, sentencepiece
```


### 3. Using with PyTorch and Transformer from Hugging Face

- Example : Sentence to Vector

```python
import torch
from noun_splitter import NounSplitter
from transformers import AlbertTokenizer, AlbertModel

# Load noun-splitter 
noun_splitter = NounSplitter("./model/np2.crfsuite")

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AlbertTokenizer.from_pretrained('model-path')

# Load pre-trained model
kb_albert = AlbertModel.from_pretrained('model-path')

# Tokenize inputs
text = "나는 국민은행에서 오픈한 알버트를 쓴다."
tokenized_text = noun_splitter.do_split(text)
input_ids = tokenizer.encode(tokenized_text)

# Convert inputs to PyTorch tensors
input_tensors = torch.tensor([input_ids])

# Predict hidden states features for each layer
with torch.no_grad():
    outputs = kb_albert(input_tensors)
    last_layer = outputs[0]
```

## Sub-tasks

|도메인|테스크(데이터셋)|Bert base multi-lingual|KB-ALBERT|
|---|---|---|---|
|일반|감성분류(Naver)|0.888|0.91|
|일반|MRC(KorQuAD 1.0)|0.87|0.90|
|금융|MRC(자체)|0.77|0.89|
