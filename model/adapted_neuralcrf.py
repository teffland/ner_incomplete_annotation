#
# @author: Allan
#

import torch
import torch.nn as nn

from config import START, STOP, PAD, log_sum_exp_pytorch

# from model.charbilstm import CharBiLSTM
# from model.bilstm_encoder import BiLSTMEncoder
from transformers import AutoModelForTokenClassification
from model.adapted_linear_crf_inferencer import LinearCRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import ContextEmb
from typing import Tuple
from overrides import overrides


class NNCRF(nn.Module):
    def __init__(self, config, print_info: bool = True):
        super(NNCRF, self).__init__()
        self.device = config.device
        # self.encoder = BiLSTMEncoder(config, print_info=print_info)
        self.encoder = AutoModelForTokenClassification.from_pretrained(config.bert_model, num_labels=config.label_size)
        self.encoder.to(self.device)
        self.inferencer = LinearCRF(config, print_info=print_info)
        self.inferencer.to(self.device)

    @overrides
    def forward(
        self,
        words: torch.Tensor,
        word_seq_lens: torch.Tensor,
        # batch_context_emb: torch.Tensor,
        # chars: torch.Tensor,
        # char_seq_lens: torch.Tensor,
        annotation_mask: torch.Tensor,
        marginals: torch.Tensor,
        tags: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the negative loglikelihood.
        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        # :param batch_context_emb: (batch_size x max_seq_len x context_emb_size)
        # :param chars: (batch_size x max_seq_len x max_char_len)
        # :param char_seq_lens: (batch_size x max_seq_len)
        :param tags: (batch_size x max_seq_len)
        :return: the loss with shape (batch_size)
        """
        batch_size = words.size(0)
        sent_len = words.size(1)
        maskTemp = (
            torch.arange(1, sent_len + 1, dtype=torch.long)
            .view(1, sent_len)
            .expand(batch_size, sent_len)
            .to(self.device)
        )
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device)
        # attn_mask = torch.zeros_like(words).float()
        # for i, l in enumerate(word_seq_lens):
        #     print(i, l)
        #     attn_mask[i, :l] = 1.0
        # print("mask", mask.shape)  # , mask)
        bert_scores = self.encoder(words, mask)[0]
        # print("bert scores", bert_scores.shape)  # , bert_scores)
        unlabed_score, labeled_score = self.inferencer(bert_scores, word_seq_lens, tags, mask)
        return unlabed_score - labeled_score

    def decode(self, batchInput: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        (
            wordSeqTensor,
            wordSeqLengths,
            # batch_context_emb,
            # charSeqTensor,
            # charSeqLengths,
            annotation_mask,
            marginals,
            tagSeqTensor,
        ) = batchInput
        batch_size = wordSeqTensor.size(0)
        sent_len = wordSeqTensor.size(1)
        maskTemp = (
            torch.arange(1, sent_len + 1, dtype=torch.long)
            .view(1, sent_len)
            .expand(batch_size, sent_len)
            .to(self.device)
        )
        mask = torch.le(maskTemp, wordSeqLengths.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device)
        features = self.encoder(wordSeqTensor, mask)[0]
        bestScores, decodeIdx = self.inferencer.decode(features, wordSeqLengths, annotation_mask)
        return bestScores, decodeIdx

    def get_marginal(self, batchInput: Tuple) -> torch.Tensor:
        (
            wordSeqTensor,
            wordSeqLengths,
            # batch_context_emb,
            # charSeqTensor,
            # charSeqLengths,
            annotation_mask,
            marginals,
            tagSeqTensor,
        ) = batchInput
        batch_size = wordSeqTensor.size(0)
        sent_len = wordSeqTensor.size(1)
        maskTemp = (
            torch.arange(1, sent_len + 1, dtype=torch.long)
            .view(1, sent_len)
            .expand(batch_size, sent_len)
            .to(self.device)
        )
        mask = torch.le(maskTemp, wordSeqLengths.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device)
        features = self.encoder(wordSeqTensor, mask)[0]
        marginals = self.inferencer.compute_constrained_marginal(features, wordSeqLengths, annotation_mask)
        return marginals