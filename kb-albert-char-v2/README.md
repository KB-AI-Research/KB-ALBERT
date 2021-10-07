# KB-ALBERT-CHAR-v2

Character-level KB-ALBERT Model and Tokenizer (Version 2)

## v1 과의 차이점

- Transformers의 `AutoTokenizer` (혹은 `BertTokenizer`) 를 통해 바로 사용 가능

  ```python
  >>> from transformers import AutoTokenizer
  >>> tokenizer = AutoTokenizer.from_pretrained(kb_albert_model_path)
  >>> tokenizer.tokenize("KB-ALBERT의 음절단위 토크나이저입니다.")
  ['K', '##B', '-', 'A', '##L', '##B', '##E', '##R', '##T', '##의', '음', '##절', '##단', '##위', '토', '##크', '##나', '##이', '##저', '##입', '##니', '##다', '.']
  ```

- `vocab size`를 23797 -> **9607**로 줄임
- `학습 데이터`의 경우 40GB -> **100GB**로 늘려서 학습

</br>

## How to use

### 1. Model Download

- KB-ALBERT를 사용하시고자 하시는 분들은 아래 메일로 소속, 이름, 사용용도를 간단히 작성하셔서 발송해 주세요.
- ai.kbg@kbfg.com

### 2. Requirements

- `torch>=1.4.1`
- `transformers>=3.5.1,<5.0.0`

```bash
pip install -r requirements.txt
```

### 3. Unzip model zip file

- 메일로 제공받은 압축파일들을 해제합니다.

```bash
unzip kb-albert-char-base-v2.zip
```

### 4. Using with Transformers from Hugging Face

- PyTorch

  ```python
  from transformers import AutoModel, AutoTokenizer

  # Load Tokenizer and Model
  kb_albert_model_path = "kb-albert-char-base-v2"
  pt_model = AutoModel.from_pretrained(kb_albert_model_path)
  tokenizer = AutoTokenizer.from_pretrained(kb_albert_model_path)

  # inference text input to sentence vector of last layer
  text = '방카슈랑스는 금융의 겸업화 추세에 부응하여 금융산업의 선진화를 도모하고 금융소비자의 편익을 위하여 도입되었습니다.'
  pt_inputs = tokenizer(text, return_tensors='pt')
  pt_outputs = pt_model(**pt_inputs)[0]
  print(pt_outputs)

  # tensor([[[ 0.3641,  0.0061,  0.5332,  ..., -0.1811, -0.0993, -0.8858],
  #        [ 0.0622,  1.0181, -0.2391,  ...,  0.3143,  0.1828, -0.2741],
  #        [-0.0845,  0.7681,  0.5878,  ...,  0.0679, -0.8295, -0.8495],
  #        ...,
  #        [ 0.4844, -0.3906, -0.7792,  ...,  0.2099,  0.0846,  0.2969],
  #        [ 0.9396, -0.3218, -0.3431,  ...,  0.3089, -0.0844,  0.0307],
  #        [ 0.0204, -0.0686,  0.0294,  ...,  0.0107,  0.0100,  0.0119]]],
  #       grad_fn=<NativeLayerNormBackward>)
  ```

- TensorFlow 2

  > Note: `tensorflow<2.4.0` 로 설치해야 합니다. `2.4.0` 이상에서는 정상적으로 작동하지 않는 이슈가 존재합니다.

  ```python
  from transformers import TFAutoModel, AutoTokenizer

  # Load Tokenizer and Model
  kb_albert_model_path = "kb-albert-char-base-v2"
  tf_model = TFAutoModel.from_pretrained(kb_albert_model_path, from_pt=True)  # Load Model from pytorch checkpoint
  tokenizer = AutoTokenizer.from_pretrained(kb_albert_model_path)

  # inference text input to sentence vector of last layer
  text = '방카슈랑스는 금융의 겸업화 추세에 부응하여 금융산업의 선진화를 도모하고 금융소비자의 편익을 위하여 도입되었습니다.'
  tf_inputs = tokenizer(text, return_tensors='tf')
  tf_outputs = tf_model(**tf_inputs)[0]
  print(tf_outputs)

  # tf.Tensor(
  # [[[ 0.36410248  0.00608762  0.53323805 ... -0.18112431 -0.09930683
  #    -0.885822  ]
  #   [ 0.06216908  1.018093   -0.23909903 ...  0.3142932   0.18277521
  #    -0.27414644]
  #   [-0.08454809  0.7680676   0.58776677 ...  0.06786535 -0.82949126
  #    -0.84953535]
  #   ...
  #   [ 0.48439485 -0.39055535 -0.77916294 ...  0.20992328  0.08462218
  #     0.29685068]
  #   [ 0.9396333  -0.32184413 -0.34310374 ...  0.30889785 -0.08437073
  #     0.03065323]
  #   [ 0.02039097 -0.06864671  0.02943919 ...  0.01071193  0.0100258
  #     0.01190421]]], shape=(1, 54, 768), dtype=float32)
  ```

</br>

## Sub-tasks

|                         | NSMC (Acc) | KorQuAD (EM/F1) | 금융MRC (EM/F1) | Size |
| :---------------------- | :--------: | :-------------: | :-------------: | :--: |
| Bert base multi-lingual |   87.32    |  68.43 / 88.43  |  39.48 / 64.74  | 681M |
| KoBERT                  |   90.09    |  49.53 / 76.37  |  19.74 / 57.98  | 351M |
| KB-ALBERT-CHAR-v1       |   88.55    |  74.52 / 90.79  |  30.39 / 65.61  | 44M  |
| KB-ALBERT-CHAR-v2       |   89.62    |  84.14 / 92.13  |  50.12 / 68.21  | 36M  |

> Note: 테스트를 진행하는 환경에 따라 결과는 다소 차이가 있을 수 있습니다.
