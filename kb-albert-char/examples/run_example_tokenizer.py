import os
import sys
sys.path.insert(0, os.path.abspath('../'))

from pprint import pprint
from tokenization_kbalbert import KbAlbertCharTokenizer

kb_albert_model_path = '../kb-albert-model'
text = '방카슈랑스는 금융의 겸업화 추세에 부응하여 금융산업의 선진화를 도모하고 금융소비자의 편익을 위하여 도입되었습니다.'

tokenizer = KbAlbertCharTokenizer.from_pretrained(kb_albert_model_path)
pprint(tokenizer(text))

