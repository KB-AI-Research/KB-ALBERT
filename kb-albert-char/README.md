# KB-ALBERT-CHAR
Character-level KB-ALBERT Model and Tokenizer 

## 모델 상세 정보

### 1. Architecture
- max_seq_length=512
- embedding_size=128
- hidden_size=768
- num_hidden_layers=12
- vocab_size=23797


### 2. 학습 데이터 셋

- 일반 도메인 텍스트(위키 + 뉴스 등) : 약 25GB 
- 금융 도메인 텍스트(경제/금융 특화 뉴스 + 리포트 등) : 약 15GB

</br>

## Tokenizer
음절단위 한글 토크나이저
- 기본적으로 BertWordPieceTokenizer에서 음절만 있는 형태와 비슷
- 문장의 시작과 앞에의 띄어쓰기가 있는 음절을 제외하고는 음절 앞에 `"##"` prefix 추가
  <br>띄어쓰기 `" "`는 사전에서 제외
- Hugging Face의 Transformers 중 Tokenizer API를 활용하여 개발
- Transformers의 tokenization 관련 모든 기능 지원

```python
>>> from tokenization_kbalbert import KbAlbertCharTokenizer
>>> tokenizer = KbAlbertCharTokenizer.from_pretrained(kb_albert_model_path)
>>> tokenizer.tokenize("KB-ALBERT의 음절단위 토크나이저입니다.")
['K', '##B', '##-', '##A', '##L', '##B', '##E', '##R', '##T', '##의', '음', '##절', '##단', '##위', '토', '##크', '##나', '##이', '##저', '##입', '##니', '##다', '##.']
```

> Notes: 
> 1. Tokenizer는 `Transformers 3.0.x` 에서의 사용을 권장합니다. 
> 2. Tokenizer는 본 repo에서 제공하고 있는 `KbAlbertCharTokenizer`를 사용해야 합니다. (`tokenization_kbalbert.py`)
> 3. Tokenizer를 사용하기 위해서 별도로 제공된 모델이 저장된 경로에서 불러와야 합니다.

<br>

## How to use

### 1. Model Download

- KB-ALBERT를 사용하시고자 하시는 분들은 아래 메일로 소속, 이름, 사용용도를 간단히 작성하셔서 발송해 주세요.
- ai.kbg@kbfg.com

### 2. Source Download and Install

```shell script
git clone
cd kb-albert-char
pip install -r requirements.txt
```

### 3. Unzip model zip file
- 다음의 명령으로 디렉토리를 생성한 후, 해당 디렉토리에 메일로 제공받은 압축파일들을 해제합니다.
```
$ mkdir model
```

### 4. Using with Transformers from Hugging Face
추가 예제는 [링크](https://github.com/KB-Bank-AI/KB-ALBERT-KO/kb-albert-char/examples/README.md) 를 참고해주시기 바랍니다.

- For PyTorch
    ```python
    from transformers import AlbertModel
    from tokenization_kbalbert import KbAlbertCharTokenizer
    
    # Load Tokenizer and Model
    tokenizer = KbAlbertCharTokenizer.from_pretrained(kb_albert_model_path)  
    pt_model = AlbertModel.from_pretrained(kb_albert_model_path)
    
    # inference text input to sentence vector of last layer
    text = '방카슈랑스는 금융의 겸업화 추세에 부응하여 금융산업의 선진화를 도모하고 금융소비자의 편익을 위하여 도입되었습니다.'
    pt_inputs = tokenizer(text, return_tensors='pt')
    pt_outputs = pt_model(**pt_inputs)[0]
    print(pt_outputs)
    # tensor([[[-0.0488, -0.0654,  0.2096,  ..., -0.1469, -0.1098, -0.0868],
    #     [-0.1622,  0.4314,  0.1699,  ..., -0.0117, -0.1561, -0.3570],
    #     [ 0.2427,  0.1104, -0.4271,  ...,  0.2620, -0.1443,  0.0400],
    #     ...,
    #     [ 0.0707, -0.0434,  0.0327,  ..., -0.0073, -0.0551,  0.0299],
    #     [ 0.1522, -0.2932, -0.0119,  ...,  0.3564, -0.0004,  0.0474],
    #     [ 0.0707, -0.0434,  0.0327,  ..., -0.0073, -0.0551,  0.0299]]])
    ```

- For TensorFlow 2
    ```python
    from transformers import TFAlbertModel
    from tokenization_kbalbert import KbAlbertCharTokenizer
  
    # Load Tokenizer
    tokenizer = KbAlbertCharTokenizer.from_pretrained(kb_albert_model_path)
    
    # Load Model from pytorch checkpoint
    tf_model = TFAlbertModel.from_pretrained(kb_albert_model_path, from_pt=True)
  
    # Load Model from tensorflow checkpoint
    tf_model = TFAlbertModel.from_pretrained(kb_albert_model_path)
  
    # inference text input to sentence vector of last layer
    text = '방카슈랑스는 금융의 겸업화 추세에 부응하여 금융산업의 선진화를 도모하고 금융소비자의 편익을 위하여 도입되었습니다.'
    tf_inputs = tokenizer(text, return_tensors='tf')
    tf_outputs = tf_model(tf_inputs)[0]
    print(tf_outputs)
    # tf.Tensor(
    # [[[-0.04875002 -0.06537886  0.209628   ... -0.14685476 -0.1097548
    #   -0.08679993]
    #  [-0.16224587  0.4314255   0.16987738 ... -0.01173133 -0.15610015
    #   -0.35700825]
    #  [ 0.24265692  0.11041075 -0.42712831 ...  0.26199123 -0.14433491
    #    0.03997103]
    #  ...
    #  [ 0.07070741 -0.04337903  0.03268574 ... -0.00729588 -0.05506952
    #    0.02986315]
    #  [ 0.15221505 -0.29317853 -0.01190075 ...  0.3564418  -0.00044889
    #    0.04735418]
    #  [ 0.07071128 -0.04337839  0.03268231 ... -0.00730597 -0.0550781
    #    0.02987067]]], shape=(1, 54, 768), dtype=float32)
    ```

##  Sub-tasks
|                         | NSMC (Acc) | KorQuAD (EM/F1) | 금융MRC (EM/F1) | Size |
| ----------------------- | ---------- | --------------- | -------------- | ---- |
| Bert base multi-lingual | 86.38      | 67.63 / 87.51   | 35.56 / 60.46  | 681M |
| KoBERT                  | 89.36      | 47.99 / 74.86   | 17.14 / 59.07  | 351M |
| KB-ALBERT-CHAR          | 88.49      | 75.04 / 91.08   | 74.73 / 81.50  |  44M |
    
> Note: 테스트를 진행하는 환경에 따라 결과는 다소 차이가 있을 수 있습니다.
