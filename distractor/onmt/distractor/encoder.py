"""Define encoder for distractor generation."""
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from onmt.utils.rnn_factory import rnn_factory
from onmt.utils.misc import sequence_mask


class PermutationWrapper:
    """Sort the batch according to length, for using RNN pack/unpack"""
    def __init__(self, inputs, length, batch_first=False, rnn_type='LSTM'):
        """
        :param inputs: [seq_length * batch_size]
        :param length: [each sequence length]
        :param batch_first: the first dimension of inputs denotes batch_size or not
        """
        if batch_first:
            inputs = torch.transpose(inputs, 0, 1)
        self.original_inputs = inputs
        self.original_length = length
        self.rnn_type = rnn_type
        self.sorted_inputs = []
        self.sorted_length = []
        # store original position in mapping,
        # e.g. mapping[1] = 5 denotes the tensor which currently in self.sorted_inputs position 5
        # originally locates in position 1 of self.original_inputs
        self.mapping = torch.zeros(self.original_length.size(0)).long().fill_(0)


    def sort(self):
        """
        sort the inputs according to length
        :return: sorted tensor and sorted length
        """
        inputs_list = list(inputs_i.squeeze(1) for inputs_i
                           in torch.split(self.original_inputs, 1, dim=1))
        sorted_inputs = sorted([(length_i.item(), i, inputs_i) for i, (length_i, inputs_i) in
                                enumerate(zip(self.original_length, inputs_list))], reverse=True)
        for i, (length_i, original_idx, inputs_i) in enumerate(sorted_inputs):
            # original_idx: original position in the inputs
            self.mapping[original_idx] = i
            self.sorted_inputs.append(inputs_i)
            self.sorted_length.append(length_i)
        rnn_inputs = torch.stack(self.sorted_inputs, dim=1)
        rnn_length = torch.Tensor(self.sorted_length).type_as(self.original_length)
        return rnn_inputs, rnn_length


    def remap(self, output, state):
        """
        remap the output from RNN to the original input order
        :param output: output from nn.LSTM/GRU, all hidden states
        :param state: final state
        :return: the output and states in original input order
        """
        if self.rnn_type=='LSTM':
            remap_state = tuple(torch.index_select(state_i, 1, self.mapping.cuda())
                                for state_i in state)
        else:
            remap_state = torch.index_select(state, 1, self.mapping.cuda())

        remap_output = torch.index_select(output, 1, self.mapping.cuda())

        return remap_output, remap_state


class PermutationWrapper2D:
    """Permutation Wrapper for 2 levels input like sentence level/word level"""
    def __init__(self, inputs, word_length, sentence_length,
                 batch_first=False, rnn_type='LSTM'):
        """
        :param inputs: 3D input, [batch_size, sentence_seq_length, word_seq_length]
        :param word_length: number of tokens in each sentence
        :param sentence_length: number of sentences in each sample
        :param batch_first: batch_size in first dim of inputs or not
        :param rnn_type: LSTM/GRU
        """
        if batch_first:
            batch_size = inputs.size(0)
        else:
            batch_size = inputs.size(-1)
        self.batch_first = batch_first
        self.original_inputs = inputs
        self.original_word_length = word_length
        self.original_sentence_length = sentence_length
        self.rnn_type = rnn_type
        self.sorted_inputs = []
        self.sorted_length = []
        # store original position in mapping,
        # e.g. mapping[1][3] = 5 denotes the tensor which currently
        # in self.sorted_inputs position 5
        # originally locates in position [1][3] of self.original_inputs
        self.mapping = torch.zeros(batch_size,
                                   sentence_length.max().item()).long().fill_(0)  # (batch_n,sent_n)

    def sort(self):
        """
        sort the inputs according to length
        :return: sorted tensor and sorted length, effective_batch_size: true number of batches
        """
        # first reshape the src into a nested list, remove padded sentences
        inputs_list = list(inputs_i.squeeze(0) for inputs_i
                           in torch.split(self.original_inputs, 1, 0))
        inputs_nested_list = []
        for sent_len_i, sent_i in zip(self.original_sentence_length, inputs_list):
            sent_tmp = list(words_i.squeeze(0) for words_i in torch.split(sent_i, 1, 0))
            inputs_nested_list.append(sent_tmp[:sent_len_i])
        # remove 0 in word_length
        inputs_length_nested_list = []
        for sent_len_i, word_len_i in zip(self.original_sentence_length,
                                          self.original_word_length):
            inputs_length_nested_list.append(word_len_i[:sent_len_i])
        # get a orderedlist, each element: (word_len, sent_idx, word_ijdx, words)
        # sent_idx: i_th example in the batch
        # word_ijdx: j_th sentence sequence in the i_th example
        sorted_inputs = sorted([(sent_len_i[ij].item(), i, ij, word_ij)
                                for i, (sent_i, sent_len_i) in
                                enumerate(zip(inputs_nested_list, inputs_length_nested_list))
                                for ij, word_ij in enumerate(sent_i)], reverse=True)
        # sorted output
        rnn_inputs = []
        rnn_length = []
        for i, word_ij in enumerate(sorted_inputs):
            len_ij, ex_i, sent_ij, words_ij = word_ij
            self.mapping[ex_i, sent_ij] = i + 1  # i+1 because 0 is for empty.
            rnn_inputs.append(words_ij)
            rnn_length.append(len_ij)
        effective_batch_size = len(rnn_inputs)
        rnn_inputs = torch.stack(rnn_inputs, dim=1)
        rnn_length = torch.Tensor(rnn_length).type_as(self.original_word_length)

        return rnn_inputs, rnn_length, effective_batch_size

    def remap(self, output, state):
        """
        remap the output from RNN to the original input order
        :param output: output from nn.LSTM/GRU, all hidden states
        :param state: final state
        :return: the output and states in original input order
        here the returned batch is original_batch * max_len_sent, we need to reshape it further
        """
        # add a all_zero example at the first place
        output_padded = F.pad(output, (0, 0, 1, 0))
        remap_output = torch.index_select(output_padded, 1, self.mapping.view(-1).cuda())
        remap_output = remap_output.view(remap_output.size(0),
                                         self.mapping.size(0), self.mapping.size(1), -1)

        if self.rnn_type == "LSTM":
            h, c = state[0], state[1]
            h_padded = F.pad(h, (0, 0, 1, 0))
            c_padded = F.pad(c, (0, 0, 1, 0))
            remap_h = torch.index_select(h_padded, 1, self.mapping.view(-1).cuda())
            remap_c = torch.index_select(c_padded, 1, self.mapping.view(-1).cuda())
            remap_state = (remap_h.view(remap_h.size(0), self.mapping.size(0), self.mapping.size(1), -1),
                           remap_c.view(remap_c.size(0), self.mapping.size(0), self.mapping.size(1), -1))
        else:
            state_padded = F.pad(state, (0, 0, 1, 0))
            remap_state = torch.index_select(state_padded, 1, self.mapping.view(-1).cuda())
            remap_state = remap_state.view(remap_state.size(0), self.mapping.size(0), self.mapping.size(1), -1)
        return remap_output, remap_state


class RNNEncoder(nn.Module):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, emb_size=300):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=emb_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

    def forward(self, src_emb, lengths=None):
        "See :obj:`EncoderBase.forward()`"
        packed_emb = src_emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(src_emb, lengths_list)

        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]
        # here encoder_final[0][2,:,:] is the upper layer forward representation
        # here encoder_final[0][3,:,:] is the upper layer backward representation
        return memory_bank, encoder_final, lengths


class DistractorEncoder(nn.Module):
    """
    Distractor Generation Encoder
    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type,
                 word_encoder_type,  sent_encoder_type, question_init_type,
                 word_encoder_layers, sent_encoder_layers, question_init_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 l_ques=0.0, l_ans=0.0):
        super(DistractorEncoder, self).__init__()
        assert embeddings is not None
        self.rnn_type = rnn_type
        self.embeddings = embeddings
        self.l_ques = l_ques
        self.l_ans = l_ans

        # word encoder
        if word_encoder_type in ['brnn', 'rnn']:
            word_bidirectional = True if word_encoder_type=='brnn' else False
            word_dropout =  0.0 if word_encoder_layers == 1 else dropout
            self.word_encoder = RNNEncoder(rnn_type, word_bidirectional,
                                           word_encoder_layers, hidden_size,
                                           word_dropout, self.embeddings.embedding_size)
        else:
            raise NotImplementedError

        # sent encoder
        if sent_encoder_type in ['brnn', 'rnn']:
            sent_bidirectional = True if sent_encoder_type == 'brnn' else False
            sent_dropout = 0.0 if sent_encoder_layers == 1 else dropout
            self.sent_encoder = RNNEncoder(rnn_type, sent_bidirectional,
                                           sent_encoder_layers, hidden_size,
                                           sent_dropout, hidden_size)
        else:
            raise NotImplementedError

        # decoder hidden state initialization
        # here only use a unidirectional rnn to encode question
        if question_init_type in ['brnn', 'rnn']:
            init_bidirectional = True if question_init_type == 'brnn' else False
            ques_dropout = 0.0 if question_init_layers == 1 else dropout
            self.init_encoder = RNNEncoder(rnn_type, init_bidirectional,
                                           question_init_layers, hidden_size,
                                           ques_dropout, self.embeddings.embedding_size)
        else:
            raise NotImplementedError

        # static attention
        self.match_linear = nn.Linear(hidden_size, hidden_size)
        self.norm_linear = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def score(self, h_t, h_s):
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_dim = h_t.size()

        h_t_ = self.match_linear(h_t)
        h_t = h_t_.view(tgt_batch, 1, tgt_dim)
        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_)

    def forward(self, src, ques, ans, sent_length, word_length, ques_length, ans_length):

        wrapped_word = PermutationWrapper2D(src, word_length, sent_length, batch_first=True, rnn_type=self.rnn_type)
        sorted_word, sorted_word_length, sorted_bs = wrapped_word.sort()  # sort
        # [feat, bs, len] -> [len, bs, feat]
        sorted_word_emb = self.embeddings(sorted_word.unsqueeze(-1))  # get embedding
        sorted_word_bank, sorted_word_state, _ = self.word_encoder(sorted_word_emb, sorted_word_length)
        word_bank, word_state = wrapped_word.remap(sorted_word_bank, sorted_word_state)

        ## sentence level
        _, bs, sentlen, hid = word_state[0].size()
        # [2n, bs, sentlen, hid] -> [sentlen, bs, 2n, hid] -> [sentlen, bs, hid * 2n]
        sent_emb = word_state[0].transpose(0, 2)[:, :, -2:, :].contiguous().view(sentlen, bs, -1)
        wrapped_sent = PermutationWrapper(sent_emb, sent_length, rnn_type=self.rnn_type)
        sorted_sent_emb, sorted_sent_length = wrapped_sent.sort()
        sorted_sent_bank, sorted_sent_state, _ = self.sent_encoder(sorted_sent_emb, sorted_sent_length)
        sent_bank, sent_state = wrapped_sent.remap(sorted_sent_bank, sorted_sent_state)

        # question initializer
        wrapped_quesinit = PermutationWrapper(ques, ques_length, rnn_type=self.rnn_type)
        sorted_quesinit, sorted_quesinit_length = wrapped_quesinit.sort()  # sort
        sorted_quesinit_emb = self.embeddings(sorted_quesinit.unsqueeze(-1))  # get embedding
        sorted_quesinit_bank, sorted_quesinit_state, _ = self.init_encoder(sorted_quesinit_emb, sorted_quesinit_length)  # encode
        quesinit_bank, quesinit_state = wrapped_quesinit.remap(sorted_quesinit_bank, sorted_quesinit_state)  # remap

        # answer
        wrapped_ans = PermutationWrapper(ans, ans_length, rnn_type=self.rnn_type)
        sorted_ans, sorted_ans_length = wrapped_ans.sort()  # sort
        sorted_ans_emb = self.embeddings(sorted_ans.unsqueeze(-1))  # get embedding
        sorted_ans_bank, sorted_ans_state, _ = self.word_encoder(sorted_ans_emb, sorted_ans_length)  # encode
        ans_bank, ans_state = wrapped_ans.remap(sorted_ans_bank, sorted_ans_state)  # remap

        # question
        wrapped_ques = PermutationWrapper(ques, ques_length, rnn_type=self.rnn_type)
        sorted_ques, sorted_ques_length = wrapped_ques.sort()  # sort
        sorted_ques_emb = self.embeddings(sorted_ques.unsqueeze(-1))  # get embedding
        sorted_ques_bank, sorted_ques_state, _ = self.word_encoder(sorted_ques_emb, sorted_ques_length)  # encode
        ques_bank, ques_state = wrapped_ques.remap(sorted_ques_bank, sorted_ques_state)  # remap

        # static attention
        match_ans = torch.div(ans_bank.sum(0), ans_length.unsqueeze(-1).float() + 1e-20)
        match_ques = torch.div(ques_bank.sum(0), (ques_length.unsqueeze(-1).float() + 1e-20))
        match_word = torch.div(word_bank.sum(0), (word_length.unsqueeze(-1).float() + 1e-20))
        match_score = self.l_ques * self.score(match_ques, match_word) - self.l_ans * self.score(match_ans, match_word)
        match_score = match_score.squeeze(1)
        temperature = self.sigmoid(self.norm_linear(match_ques)) + 1e-20
        static_attn = torch.div(match_score, temperature) + 1e-20
        return word_bank, sent_bank, quesinit_state, static_attn


