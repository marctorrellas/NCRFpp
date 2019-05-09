# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-12-04 23:19:38
# @Last Modified by:   Marc Torrellas Socastro,     Contact: marc.torsoc@gmail.com
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(
        -1, 1, m_size
    )  # B * M
    tmp = torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(
        -1, m_size
    )  # B * M
    return tmp + max_score.view(-1, m_size)


class CRF(nn.Module):
    def __init__(self, tagset_size, gpu, padding_tag=0, start_tag=-2, stop_tag=-1):
        """

        :param tagset_size (int): #tags that can be predicted + padding
            the unknown and padding tag are allocated to the same index
            start and stop will be added to this
        :param gpu (boolean): whether to use GPU
        :param padding_tag (int): index for the padding tag (same as unknown tag)
        :param start_tag (int): index for the start tag
        :param stop_tag (int): index for the stop tag
        """
        super(CRF, self).__init__()
        logger.info("build CRF...")
        self.gpu = gpu
        # Matrix of transition parameters.  Entry i,j is the score of transitioning
        # from i to j.
        self.tagset_size = tagset_size
        # # We add 2 here, because of START_TAG and STOP_TAG
        # # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        init_transitions = torch.zeros(self.tagset_size + 2, self.tagset_size + 2)
        self.padding_tag = padding_tag
        self.start_tag = start_tag
        self.stop_tag = stop_tag
        # we don't want any tag to go to start_tag
        init_transitions[:, self.start_tag] = -10000.0
        # we don't want any tag to come from stop_tag
        init_transitions[self.stop_tag, :] = -10000.0
        # make the unknown_tag / padding_tag impossible to predict
        init_transitions[:, self.padding_tag] = -10000.0
        init_transitions[self.padding_tag, :] = -10000.0
        if self.gpu:
            init_transitions = init_transitions.cuda()
        self.transitions = nn.Parameter(init_transitions)

    def _calculate_pz(self, feats, mask):
        """
            input:
                feats: (batch_size, seq_len, self.tag_size+2)
                masks: (batch_size, seq_len)

            feats[sample n<batch_size, time t<seq_len, tag yt<(tag_size+2)]
            tag[0]=UNKNOWN, tag[-1]=START, tag[-2]=STOP
        """
        batch_size, seq_len, tag_size = feats.size()
        assert tag_size == self.tagset_size + 2
        # swap seq_len and batch dimensions
        feats = feats.transpose(1, 0).contiguous()
        mask = mask.transpose(1, 0).contiguous()
        # prepare feats for vectorized operations
        ins_num = seq_len * batch_size
        feats = feats.view(ins_num, 1, tag_size)
        feats = feats.expand(ins_num, tag_size, tag_size)
        # scores[t,n,yt-1,yt] = feats[n,t,yt] + transitions[yt-1, yt]
        # this sum comes from separating two types of feature templates:
        #    i) input features f(yt, x) --> here outputs of the LSTM
        #    ii) transition features f(yt-1, yt) --> the only ones learned here
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(
            ins_num, tag_size, tag_size
        )
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # we will iterate for each time slot
        seq_iter = enumerate(scores)
        # scores_0 = scores[t=0]
        _, scores_0 = next(seq_iter)
        # alpha[n, yt] = scores[t=0, n, yt-1=START_TAG, yt]
        alpha = scores_0[:, self.start_tag, :].clone().view(batch_size, tag_size, 1)

        # for each timestemp iterate over scores
        for t, score_t in seq_iter:
            # score_t: (batch_size, from_target, to_target)
            #       or (sample,tag_size, tag_size)
            # alpha: (batch_size, tag_size)

            # Get log_alpha as log_sum_exp(scores + prev_log_alpha)
            tmp = alpha.contiguous().view(batch_size, tag_size, 1)
            tmp = tmp.expand(batch_size, tag_size, tag_size)
            # TODO: for some reason, we cannot do += with tensors, it gives a different result
            score_t = score_t + tmp
            log_alpha = log_sum_exp(score_t, tag_size)
            # log_alpha is a tensor (batch_size, tag_size)
            # Get mask for this timestamp into (batch_size, tag_size) tensor
            mask_idx = mask[t, :].view(batch_size, 1).expand(batch_size, tag_size)
            # Select values in alpha where mask=1
            masked_alpha = log_alpha.masked_select(mask_idx)
            # Make mask_idx broadcastable, to disable warning
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
            # Replace alpha values where mask=1, other values untouched
            alpha.masked_scatter_(mask_idx, masked_alpha)

        # until the last state, add transition score for all alpha
        # (and do log_sum_exp) then select the value in STOP_TAG
        tmp1 = self.transitions.view(1, tag_size, tag_size)
        tmp1 = tmp1.expand(batch_size, tag_size, tag_size)
        tmp2 = alpha.contiguous().view(batch_size, tag_size, 1)
        tmp2 = tmp2.expand(batch_size, tag_size, tag_size)
        score_t = tmp1 + tmp2
        log_alpha = log_sum_exp(score_t, tag_size)
        # finishing at STOP_TAG, sum for all samples in batch
        return log_alpha[:, self.stop_tag].sum(), scores

    def viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implemented)
        """
        # TODO: clean this up
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert tag_size == self.tagset_size + 2
        # calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        # mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()
        # be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        ins_num = seq_len * batch_size
        feats = (
            feats.transpose(1, 0)
            .contiguous()
            .view(ins_num, 1, tag_size)
            .expand(ins_num, tag_size, tag_size)
        )
        # need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(
            ins_num, tag_size, tag_size
        )
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        # record the position of best score
        back_points = list()
        partition_history = list()
        #  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = (
            inivalues[:, self.start_tag, :].clone().view(batch_size, tag_size)
        )  # bat_size * to_target_size
        partition_history.append(partition)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: batch_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1
            ).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            # cur_bp: (batch_size, tag_size) max source score position in current tag
            # set padded label as 0, which will be filtered in post processing
            cur_bp.masked_fill_(
                mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0
            )
            back_points.append(cur_bp)

        # add score to final STOP_TAG
        partition_history = (
            torch.cat(partition_history, 0)
            .view(seq_len, batch_size, -1)
            .transpose(1, 0)
            .contiguous()
        )  # (batch_size, seq_len. tag_size)
        # get the last position for each setences, and select the last partitions using gather()
        last_position = (
            length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        )
        last_partition = torch.gather(partition_history, 1, last_position).view(
            batch_size, tag_size, 1
        )
        # calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(
            batch_size, tag_size, tag_size
        ) + self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size
        )
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        # select end ids in STOP_TAG
        pointer = last_bp[:, self.stop_tag]
        insert_last = (
            pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        )
        back_points = back_points.transpose(1, 0).contiguous()
        # move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1, 0).contiguous()
        # decode from the end, padded position ids are 0, which will be filtered if following evaluation
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if self.gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.detach()
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(
                back_points[idx], 1, pointer.contiguous().view(batch_size, 1)
            )
            decode_idx[idx] = pointer.detach().view(batch_size)
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, feats):
        path_score, best_path = self.viterbi_decode(feats)
        return path_score, best_path

    def _score_sentence(self, scores, mask, tags):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        seq_len, batch_size, tag_size, _ = scores.size()
        # convert tag value into a new format, recorded label bigram information to index
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if self.gpu:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                # start -> first score
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]

            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]

        # transition for label to STOP_TAG
        end_transition = self.transitions[:, self.stop_tag].contiguous()
        end_transition = end_transition.view(1, tag_size).expand(batch_size, tag_size)
        # length for batch,  last word position = length - 1
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        # index the label id of last word
        end_ids = torch.gather(tags, 1, length_mask - 1)

        # index the transition score for end_id to STOP_TAG
        end_energy = torch.gather(end_transition, 1, end_ids)

        # convert tag as (seq_len, batch_size, 1)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        # need convert tags id to search from 400 positions of scores, seq_len * bat_size
        tg_energy = torch.gather(
            scores.view(seq_len, batch_size, -1), 2, new_tags
        ).view(seq_len, batch_size)
        # mask transpose to (seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        # add all scores together
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        forward_score, scores = self._calculate_pz(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        return forward_score - gold_score

    def viterbi_decode_nbest(self, feats, mask, nbest):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, nbest, seq_len) decoded sequence
                path_score: (batch, nbest) corresponding score for each sequence (to be implementated)
                nbest decode for sentence with one token is not well supported, to be optimized
        """
        # TODO: clean this up
        batch_size, seq_len, tag_size = feats.size()
        assert tag_size == self.tagset_size + 2
        # calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        # mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        # be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = (
            feats.transpose(1, 0)
            .contiguous()
            .view(ins_num, 1, tag_size)
            .expand(ins_num, tag_size, tag_size)
        )
        # need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(
            ins_num, tag_size, tag_size
        )
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        # record the position of best score
        back_points = list()
        partition_history = list()
        #  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_tag, :].clone()  # bat_size * to_target_size
        # initial partition [batch_size, tag_size]
        partition_history.append(
            partition.view(batch_size, tag_size, 1).expand(batch_size, tag_size, nbest)
        )
        # iter over last scores
        for idx, cur_values in seq_iter:
            if idx == 1:
                cur_values = cur_values.view(
                    batch_size, tag_size, tag_size
                ) + partition.contiguous().view(batch_size, tag_size, 1).expand(
                    batch_size, tag_size, tag_size
                )
            else:
                # previous to_target is current from_target
                # partition: previous results log(exp(from_target)), #(batch_size * nbest * from_target)
                # cur_values: batch_size * from_target * to_target
                cur_values = cur_values.view(batch_size, tag_size, 1, tag_size).expand(
                    batch_size, tag_size, nbest, tag_size
                ) + partition.contiguous().view(batch_size, tag_size, nbest, 1).expand(
                    batch_size, tag_size, nbest, tag_size
                )
                # compare all nbest and all from target
                cur_values = cur_values.view(batch_size, tag_size * nbest, tag_size)
            partition, cur_bp = torch.topk(cur_values, nbest, 1)
            # cur_bp/partition: [batch_size, nbest, tag_size], id should be normize through nbest in following backtrace step
            if idx == 1:
                cur_bp = cur_bp * nbest
            partition = partition.transpose(2, 1)
            cur_bp = cur_bp.transpose(2, 1)

            # partition: (batch_size * to_target * nbest)
            # cur_bp: (batch_size * to_target * nbest) Notice the cur_bp number is the whole position of tag_size*nbest, need to convert when decode
            partition_history.append(partition)
            # cur_bp: (batch_size,nbest, tag_size) topn source score position in current tag
            # set padded label as 0, which will be filtered in post processing
            # mask[idx] ? mask[idx-1]
            cur_bp.masked_fill_(
                mask[idx].view(batch_size, 1, 1).expand(batch_size, tag_size, nbest), 0
            )
            back_points.append(cur_bp)
        # add score to final STOP_TAG
        partition_history = (
            torch.cat(partition_history, 0)
            .view(seq_len, batch_size, tag_size, nbest)
            .transpose(1, 0)
            .contiguous()
        )  # (batch_size, seq_len, nbest, tag_size)
        # get the last position for each setences, and select the last partitions using gather()
        last_position = (
            length_mask.view(batch_size, 1, 1, 1).expand(batch_size, 1, tag_size, nbest)
            - 1
        )
        last_partition = torch.gather(partition_history, 1, last_position).view(
            batch_size, tag_size, nbest, 1
        )
        # calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(
            batch_size, tag_size, nbest, tag_size
        ) + self.transitions.view(1, tag_size, 1, tag_size).expand(
            batch_size, tag_size, nbest, tag_size
        )
        last_values = last_values.view(batch_size, tag_size * nbest, tag_size)
        end_partition, end_bp = torch.topk(last_values, nbest, 1)
        # end_partition: (batch, nbest, tag_size)
        end_bp = end_bp.transpose(2, 1)
        # end_bp: (batch, tag_size, nbest)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size, nbest)).long()
        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size, nbest)

        # select end ids in STOP_TAG
        pointer = end_bp[:, self.stop_tag, :]  # (batch_size, nbest)
        insert_last = (
            pointer.contiguous()
            .view(batch_size, 1, 1, nbest)
            .expand(batch_size, 1, tag_size, nbest)
        )
        back_points = back_points.transpose(1, 0).contiguous()
        # move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        # copy the ids of last position:insert_last to back_points, though the last_position index
        # last_position includes the length of batch sentences
        back_points.scatter_(1, last_position, insert_last)
        # back_points: [batch_size, seq_length, tag_size, nbest]
        """
        back_points: in simple demonstration
        x,x,x,x,x,x,x,x,x,7
        x,x,x,x,x,4,0,0,0,0
        x,x,6,0,0,0,0,0,0,0
        """

        back_points = back_points.transpose(1, 0).contiguous()
        # back_points: (seq_len, batch, tag_size, nbest)
        # decode from the end, padded position ids are 0, which will be filtered in following evaluation
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size, nbest))
        if self.gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data / nbest
        # use old mask, let 0 means has token
        for idx in range(len(back_points) - 2, -1, -1):
            new_pointer = torch.gather(
                back_points[idx].view(batch_size, tag_size * nbest),
                1,
                pointer.contiguous().view(batch_size, nbest),
            )
            decode_idx[idx] = new_pointer.data / nbest
            # # use new pointer to remember the last end nbest ids for non longest
            pointer = (
                new_pointer
                + pointer.contiguous().view(batch_size, nbest)
                * mask[idx].view(batch_size, 1).expand(batch_size, nbest).long()
            )

        decode_idx = decode_idx.transpose(1, 0)
        # decode_idx: [batch, seq_len, nbest]

        # calculate probability for each sequence
        scores = end_partition[:, :, self.stop_tag]
        # scores: [batch_size, nbest]
        max_scores, _ = torch.max(scores, 1)
        minus_scores = scores - max_scores.view(batch_size, 1).expand(batch_size, nbest)
        path_score = F.softmax(minus_scores, 1)
        # path_score: [batch_size, nbest]
        return path_score, decode_idx
