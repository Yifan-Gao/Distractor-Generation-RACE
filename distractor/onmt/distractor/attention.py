""" Hierarchical attention modules """
import torch
import torch.nn as nn

from onmt.utils.misc import aeq, sequence_mask, sequence_mask_herd


class HierarchicalAttention(nn.Module):
    """Dynamic attention"""
    def __init__(self, dim, attn_type="general"):
        super(HierarchicalAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
            "Please select a valid attention type.")

        # Hierarchical attention
        if self.attn_type == "general":
            self.word_linear_in = nn.Linear(dim, dim, bias=False)
            self.sent_linear_in = nn.Linear(dim, dim, bias=False)
        else:
            raise NotImplementedError

        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def score(self, h_t, h_s, type):
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_dim = h_t.size()
        if type == 'word':
            h_t_ = self.word_linear_in(h_t)
        elif type == 'sent':
            h_t_ = self.sent_linear_in(h_t)
        else:
            raise NotImplementedError
        h_t = h_t_.view(tgt_batch, 1, tgt_dim)
        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_)

    def forward(self, source, word_bank, word_lengths,
                sent_bank, sent_lengths, static_attn):

        # source = source.unsqueeze(1)
        word_max_len, word_batch, words_max_len, word_dim = word_bank.size()
        sent_max_len, sent_batch, sent_dim = sent_bank.size()
        assert word_batch == sent_batch
        assert words_max_len == sent_max_len
        target_batch, target_dim = source.size()

        # reshape for compute word score
        # (word_max_len, word_batch, words_max_len, word_dim) -> transpose
        # (word_batch, word_max_len, words_max_len, word_dim) -> transpose   !!! important, otherwise do not match the src_map
        # (word_batch, words_max_len, word_max_len, word_dim)
        word_bank = word_bank.contiguous().transpose(0, 1).transpose(1, 2).contiguous().view(
            word_batch, words_max_len * word_max_len, word_dim)
        word_align = self.score(source, word_bank, 'word')

        # sentence score
        # (sent_batch, target_l, sent_max_len)
        sent_bank = sent_bank.transpose(0, 1).contiguous()
        sent_align = self.score(source, sent_bank, 'sent')

        # attn
        align = (word_align.view(word_batch, 1, words_max_len, word_max_len) * sent_align.unsqueeze(-1) *\
                      static_attn.unsqueeze(1).unsqueeze(-1)).view(word_batch, 1, words_max_len * word_max_len)
        mask = sequence_mask(word_lengths.view(-1), max_len=word_max_len).view(
            word_batch, words_max_len * word_max_len).unsqueeze(1)
        align.masked_fill_(1 - mask.cuda(), -float('inf'))
        align_vectors = self.softmax(align) + 1e-20
        c = torch.bmm(align_vectors, word_bank).squeeze(1)
        concat_c = torch.cat([c, source], -1).view(target_batch, target_dim * 2)
        attn_h = self.linear_out(concat_c).view(target_batch, target_dim)
        attn_h = self.tanh(attn_h)
        return attn_h, align_vectors.squeeze(1)
