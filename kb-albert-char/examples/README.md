# Examples
본 레포를 통해 제공되는 음절단위 KB-ALBERT 모델의 사용 예제는 아래 리스트와 같습니다.
- `run_example_tokenizer.py` : 토크나이저 사용 예시
- `run_example_inference.py` : 모델 사용 예시
- `run_nsmc.py` : 네이버 영화리뷰 감성분석 모델 학습 예시
- `run_example_pipeline.py`  : 학습한 감성분석의 파이프라인 사용 예시
<br><br>
## 네이버 영화리뷰 감성분석 (Naver Sentiment Movie Review)
네이버 영화리뷰 감성분석 데이터를 활용하는 KB-ALBERT 사용 예제입니다.<br>
KB-ALBERT-CHAR 언어모델을 이용한 fine-tuning을 위한 
 - 데이터 출처: https://github.com/e9t/nsmc


### 감성분석 모델 학습
네이버 영화리뷰 감성분석 데이터는 학습 데이터 (150K)
```shell script
python run_nsmc.py ./nsmc_data/nsmc_config.json
```

### 학습한 감성분석 모델을 이용한 예측 파이프라인 개발
`run_nsmc.py`를 통해 학습한 감성분석 모델은 `run_example_pipeline.py`
```shell script
python run_example_pipeline.py
```

감성분석을 위한 파이프라인 예제는 아래와 같이 동작합니다.
```python
>>> model = AutoModelForSequenceClassification.from_pretrained(model_output_path)
>>> tokenizer = KbAlbertCharTokenizer.from_pretrained(kb_albert_model_path)

>>> nsmc_classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, framework='pt')

>>> reviews = ['이 영화 최악이었어!',
               '볼거리가 많은 내 인생 영화 ㅎㅎ']
>>> results = nsmc_classifier(reviews)
>>> for result in results:
        print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
label: negative, with score: 0.9957
label: positive, with score: 0.9541
```