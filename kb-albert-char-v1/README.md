# KB-ALBERT-CHAR-v1

Character-level KB-ALBERT Model and Tokenizer (Version 1)

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

- 기본적으로 `BertWordPieceTokenizer`에서 음절만 있는 형태와 비슷
- 문장의 시작과 앞에의 띄어쓰기가 있는 음절을 제외하고는 음절 앞에 `"##"` prefix 추가
  - 띄어쓰기 `" "`는 사전에서 제외
- Huggingface Transformers 중 Tokenizer API를 활용하여 개발
- Transformers의 tokenization 관련 모든 기능 지원

```python
>>> from tokenization_kbalbert import KbAlbertCharTokenizer
>>> tokenizer = KbAlbertCharTokenizer.from_pretrained(kb_albert_model_path)
>>> tokenizer.tokenize("KB-ALBERT의 음절단위 토크나이저입니다.")
['K', '##B', '##-', '##A', '##L', '##B', '##E', '##R', '##T', '##의', '음', '##절', '##단', '##위', '토', '##크', '##나', '##이', '##저', '##입', '##니', '##다', '##.']
```

> Notes:
>
> 1. Tokenizer는 `transformers>=3.5.1,<5.0.0` 에서의 사용을 권장합니다.
> 2. Tokenizer는 본 repo에서 제공하고 있는 `KbAlbertCharTokenizer`를 사용해야 합니다. (`tokenization_kbalbert.py`)
> 3. Tokenizer를 사용하기 위해서 별도로 제공된 모델이 저장된 경로에서 불러와야 합니다.

<br>

## How to use

### 1. Model Download

- KB-ALBERT를 사용하시고자 하시는 분들은 아래 메일로 소속, 이름, 사용용도를 간단히 작성하셔서 발송해 주세요.
- ai.kbg@kbfg.com

### 2. Source Download and Install

```bash
git clone
cd kb-albert-char
pip install -r requirements.txt
```

### 3. Unzip model zip file

- 메일로 제공받은 압축파일들을 해제합니다.

```bash
unzip kb-albert-char-base-v1.zip
```

### 4. Using with Transformers from Hugging Face

추가 예제는 [링크](./examples) 를 참고해주시기 바랍니다.

- PyTorch

  ```python
  from transformers import AlbertModel
  from tokenization_kbalbert import KbAlbertCharTokenizer

  # Load Tokenizer and Model
  kb_albert_model_path = "kb-albert-char-base-v1"
  tokenizer = KbAlbertCharTokenizer.from_pretrained(kb_albert_model_path)
  pt_model = AlbertModel.from_pretrained(kb_albert_model_path)

  # inference text input to sentence vector of last layer
  text = '방카슈랑스는 금융의 겸업화 추세에 부응하여 금융산업의 선진화를 도모하고 금융소비자의 편익을 위하여 도입되었습니다.'
  pt_inputs = tokenizer(text, return_tensors='pt')
  pt_outputs = pt_model(**pt_inputs)[0]
  print(pt_outputs)

  # tensor([[[-0.2424, -0.1150,  0.1739,  ..., -0.1104, -0.2521, -0.2343],
  #        [-0.2398,  0.6024,  0.2140,  ..., -0.1003, -0.0811, -0.3387],
  #        [-0.0628,  0.1722, -0.2954,  ...,  0.0260, -0.1288, -0.0367],
  #        ...,
  #        [ 0.0406, -0.0463,  0.0175,  ..., -0.0016, -0.0636,  0.0402],
  #        [ 0.1111, -0.2125,  0.0141,  ...,  0.1380, -0.1252, -0.0849],
  #        [ 0.0406, -0.0463,  0.0175,  ..., -0.0016, -0.0636,  0.0402]]],
  #       grad_fn=<NativeLayerNormBackward>)
  ```

- TensorFlow 2 

  > Note: `tensorflow<2.4.0` 로 설치해야 합니다. `2.4.0` 이상에서는 정상적으로 작동하지 않는 이슈가 존재합니다.

  ```python
  from transformers import TFAlbertModel
  from tokenization_kbalbert import KbAlbertCharTokenizer

  # Load Tokenizer and Model
  kb_albert_model_path = "kb-albert-char-base-v1"
  tf_model = TFAlbertModel.from_pretrained(kb_albert_model_path, from_pt=True)  # Load Model from pytorch checkpoint
  tokenizer = KbAlbertCharTokenizer.from_pretrained(kb_albert_model_path)

  # inference text input to sentence vector of last layer
  text = '방카슈랑스는 금융의 겸업화 추세에 부응하여 금융산업의 선진화를 도모하고 금융소비자의 편익을 위하여 도입되었습니다.'
  tf_inputs = tokenizer(text, return_tensors='tf')
  tf_outputs = tf_model(**tf_inputs)[0]
  print(tf_outputs)

  # tf.Tensor(
  # [[[-0.24243964 -0.11504199  0.17393394 ... -0.11044255 -0.25206143
  #    -0.23426099]
  #   [-0.23975688  0.602407    0.21395475 ... -0.10028075 -0.0811163
  #    -0.33866256]
  #   [-0.06281263  0.17218213 -0.2953698  ...  0.0259751  -0.12883013
  #    -0.03670265]
  #   ...
  #   [ 0.04058526 -0.04625401  0.01750809 ... -0.00161678 -0.06357289
  #     0.04015559]
  #   [ 0.11111069 -0.21249968  0.01409134 ...  0.13796115 -0.12516701
  #    -0.08493041]
  #   [ 0.04058542 -0.04625029  0.01748614 ... -0.0016343  -0.06360044
  #     0.0401795 ]]], shape=(1, 54, 768), dtype=float32)
  ```

</br>

## Sub-tasks

|                         | NSMC (Acc) | KorQuAD (EM/F1) | 금융MRC (EM/F1) | Size |
| :---------------------- | :--------: | :-------------: | :-------------: | :--: |
| Bert base multi-lingual |   87.32    |  68.43 / 88.43  |  39.48 / 64.74  | 681M |
| KoBERT                  |   90.09    |  49.53 / 76.37  |  19.74 / 57.98  | 351M |
| KB-ALBERT-CHAR-v1       |   88.55    |  74.52 / 90.79  |  30.39 / 65.61  | 44M  |

> Note: 테스트를 진행하는 환경에 따라 결과는 다소 차이가 있을 수 있습니다.
