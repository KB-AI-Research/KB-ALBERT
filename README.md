# KB-ALBERT

KB국민은행에서 제공하는 경제/금융 도메인에 특화된 한국어 ALBERT 언어모델

## Introduction

### KB-ALBERT 언어모델이란

- KB-ALBERT는 Google 에서 제안한 [ALBERT(A Lite BERT)](https://arxiv.org/abs/1909.11942) 아키텍처를 기반으로 대량의 한국어를 학습시킨 사전학습 언어모델입니다.
  - ALBERT는 BERT 계열의 PLM(Pretrained Language Model) 아키텍쳐의 일종으로 경량화 설계가 특징입니다.
  - 대량의 언어데이터로 학습된 언어모델이 있으면 비교적 적은 양의 학습데이터로도 Task에 맞게 finetuning 학습이 가능합니다.
- KB-ALBERT는 금융권 한국어 뿐만아니라 일반적인 한국어 Task를 학습하는 경우에도 우수한 성능을 보입니다.

  |                         | NSMC (Acc) |  KorQuAD (EM/F1)  |  금융MRC (EM/F1)  |  Size   |
  | :---------------------- | :--------: | :---------------: | :---------------: | :-----: |
  | Bert base multi-lingual |   87.32    |   68.43 / 88.43   |   39.48 / 64.74   |  681M   |
  | **KB-ALBERT-CHAR-v2**   | **89.62**  | **84.14 / 92.13** | **50.12 / 68.21** | **36M** |

### 사용 방법

- 모델의 사용법 및 예제는 하위 디렉토리에 있는 README.md 파일을 참고해주세요.
- KB-ALBERT는 **비영리**를 목적으로만 사용할 수 있습니다.
- 모델을 사용하고자 하시는 분들은 아래 메일로 **소속, 이름, 사용용도**를 간단히 작성하셔서 발송해 주세요.
  - ai.kbg@kbfg.com

## Version History

- v1 (kb-albert-char-v1) 공개
- v2 (kb-albert-char-v2) 공개

구 버전 모델 관련 코드는 이후 삭제될 수 있습니다.

## License

- KB-ALBERT의 `모델 파일`과 `코드`는 별개의 라이선스 정책이 적용됩니다.
- 이 저장소의 `코드`는 Apache-2.0 라이선스 하에 공개되어 있습니다. 라이선스 전문은 LICENSE 파일에서 확인하실 수 있습니다.
- `모델 파일`은 비영리목적으로 요청한 분들께만 개별적으로 보내드립니다. 모델 파일의 라이선스 정책은 모델과 함께 전달해드립니다.

---

(참고) 다른 곳에서 공개한 한국어 PLM도 참고해보세요.

- ETRI : KorBERT([http://aiopen.etri.re.kr/service_dataset.php](http://aiopen.etri.re.kr/service_dataset.php))
- SK T-Brain : SKT-KoBERT, SKT-KoGPT... ([https://github.com/SKTBrain](https://github.com/SKTBrain))
