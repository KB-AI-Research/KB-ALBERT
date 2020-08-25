# coding=utf-8
# Copyright 2020 Leo Kim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pycrfsuite
import os


class NounSplitter:
    __tagger: pycrfsuite.Tagger

    def __init__(
            self,
            model_file
    ):
        if not os.path.isfile(model_file):
            raise ValueError(
                "Can't find a model file at path '{}'. ".format(model_file)
            )
        self.__tagger = pycrfsuite.Tagger()
        self.__tagger.open(model_file)

    def _make_out_sentence(self, input_val, output_val):
        sent = ''
        for i in range(len(input_val)):
            sent = sent + input_val[i][0]
            if input_val[i][1] == '1' or output_val[i] == '1':
                sent = sent + ' '
        return sent.strip()

    def _sent2input(self, sent):
        input_val = []
        for i in range(len(sent)):
            if i == len(sent) - 1:
                input_val.append([sent[i], '1', '0'])
            elif sent[i + 1] == ' ':
                input_val.append([sent[i], '1', '0'])
            elif sent[i] == ' ':
                continue
            else:
                input_val.append([sent[i], '0', '0'])
        return input_val

    def _char2features(self, sent, i):
        features = {
            'bias': 1.0,
            'chars[-2]': sent[i - 2][0],
            'chars[-1]': sent[i - 1][0],
            'chars[0]': sent[i][0],
            'chars[1]': sent[i + 1][0],
            'chars[2]': sent[i + 2][0],
            'space[-2]': sent[i - 2][1],
            'space[-1]': sent[i - 1][1],
            'space[0]': sent[i][1],
            'space[1]': sent[i + 1][1],
            'space[2]': sent[i + 2][1],
            'chars[0]_space[0]': sent[i][0] + '_' + sent[i][1],
            'chars[-1]_space[-1]': sent[i - 1][0] + '_' + sent[i - 1][1],
            'chars[1]_space[1]': sent[i + 1][0] + '_' + sent[i + 1][1]
        }
        return features

    def _input2features(self, input_val):
        chars_1 = ['__S-1__', '0', '0']
        chars_2 = ['__S-2__', '0', '0']
        chars_3 = ['__S+1__', '0', '0']
        chars_4 = ['__S+2__', '0', '0']
        temp_array = [chars_2, chars_1]

        for temp in input_val:
            temp_array.append(temp)

        temp_array.append(chars_3)
        temp_array.append(chars_4)

        return [self._char2features(temp_array, i) for i in range(2, len(temp_array) - 2)]

    def do_split(self, sentence):
        input_val = self._sent2input(sentence)
        output_val = self.__tagger.tag(self._input2features(input_val))
        out_sentence = self._make_out_sentence(input_val, output_val)
        return out_sentence

    def do_split_file(self, infile, outfile):
        if not os.path.isfile(infile):
            raise ValueError(
                "Can't find a input file at path '{}'. ".format(infile)
            )
        f = open(infile, 'r', encoding='utf-8')
        o = open(outfile, 'w', encoding='utf-8')

        while True:
            line: str = f.readline()
            if not line: break
            sentence = self.do_split(line.strip())
            o.write(sentence + "\n")

        f.close()
        o.flush()
        o.close()
