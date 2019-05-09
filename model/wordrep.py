# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-01 15:52:01
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np

import logging

logger = logging.getLogger(__name__)


class WordRep(nn.Module):
    def __init__(self, data):
        super(WordRep, self).__init__()
        logger.info("build word representation...")
        self.gpu = data.HP_gpu
        self.batch_size = data.HP_batch_size
        self.sentence_classification = data.sentence_classification
        self.embedding_dim = data.word_emb_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.word_embedding = nn.Embedding(
            data.word_alphabet.size(), self.embedding_dim
        )
        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(data.pretrain_word_embedding)
            )
        else:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(
                    self.random_embedding(data.word_alphabet.size(), self.embedding_dim)
                )
            )

        self.feature_num = data.feature_num
        self.feature_embedding_dims = data.feature_emb_dims
        self.feature_embeddings = nn.ModuleList()
        for idx in range(self.feature_num):
            self.feature_embeddings.append(
                nn.Embedding(
                    data.feature_alphabets[idx].size(), self.feature_embedding_dims[idx]
                )
            )
        for idx in range(self.feature_num):
            if data.pretrain_feature_embeddings[idx] is not None:
                self.feature_embeddings[idx].weight.data.copy_(
                    torch.from_numpy(data.pretrain_feature_embeddings[idx])
                )
            else:
                self.feature_embeddings[idx].weight.data.copy_(
                    torch.from_numpy(
                        self.random_embedding(
                            data.feature_alphabets[idx].size(),
                            self.feature_embedding_dims[idx],
                        )
                    )
                )

        if self.gpu:
            self.drop = self.drop.cuda()
            self.word_embedding = self.word_embedding.cuda()
            for idx in range(self.feature_num):
                self.feature_embeddings[idx] = self.feature_embeddings[idx].cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(
                -scale, scale, [1, embedding_dim]
            )
        return pretrain_emb

    def forward(self, word_inputs, feature_inputs, word_seq_lengths):
        """
            input:
                word_inputs: (batch_size, sent_len)
                features: list [(batch_size, sent_len), (batch_len, sent_len),...]
                word_seq_lengths: list of batch_size, (batch_size,1)
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        word_embs = self.word_embedding(word_inputs)
        word_list = [word_embs]
        if not self.sentence_classification:
            for idx in range(self.feature_num):
                word_list.append(self.feature_embeddings[idx](feature_inputs[idx]))
        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent
