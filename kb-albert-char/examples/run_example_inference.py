import os
import sys
sys.path.insert(0, os.path.abspath('../'))

from transformers import AlbertModel, TFAlbertModel
from tokenization_kbalbert import KbAlbertCharTokenizer

kb_albert_model_path = '../kb-albert-model'
text = '방카슈랑스는 금융의 겸업화 추세에 부응하여 금융산업의 선진화를 도모하고 금융소비자의 편익을 위하여 도입되었습니다.'

tokenizer = KbAlbertCharTokenizer.from_pretrained(kb_albert_model_path)

# PyTorch
pt_model = AlbertModel.from_pretrained(kb_albert_model_path)
pt_inputs = tokenizer(text, return_tensors='pt')
pt_outputs = pt_model(**pt_inputs)[0]
print(pt_outputs)

# TensorFlow 2.0
tf_model = TFAlbertModel.from_pretrained(kb_albert_model_path)
tf_inputs = tokenizer(text, return_tensors='tf')
tf_outputs = tf_model(tf_inputs)[0]
print(tf_outputs)
