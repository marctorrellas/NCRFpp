# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-01-01 21:11:50
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-13 12:30:56

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wordsequence import WordSequence


class SentClassifier(nn.Module):
    def __init__(self, data):
        super(SentClassifier, self).__init__()
        print("build sentence classification network...")
        print("use_char: ", data.use_char)
        if data.use_char:
            print("char feature extractor: ", data.char_feature_extractor)
        print("word feature extractor: ", data.word_feature_extractor)

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        self.word_hidden = WordSequence(data)

    def calculate_loss(
        self, word_inputs, feature_inputs, word_seq_lengths, batch_label
    ):
        outs = self.word_hidden.sentence_representation(
            word_inputs, feature_inputs, word_seq_lengths
        )
        batch_size = word_inputs.size(0)
        outs = outs.view(batch_size, -1)
        total_loss = F.cross_entropy(outs, batch_label.view(batch_size))
        _, tag_seq = torch.max(outs, 1)
        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss, tag_seq

    def forward(self, word_inputs, feature_inputs, word_seq_lengths):
        outs = self.word_hidden.sentence_representation(
            word_inputs, feature_inputs, word_seq_lengths
        )
        batch_size = word_inputs.size(0)
        outs = outs.view(batch_size, -1)
        _, tag_seq = torch.max(outs, 1)
        return tag_seq
