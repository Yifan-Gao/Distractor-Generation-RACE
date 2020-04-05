""" memory network Model base class definition """
import torch.nn as nn

class DGModel(nn.Module):
    """
    Core trainable object in Distractor Generation. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(DGModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, ques, ans, tgt,
                sent_lengths, word_lengths, ques_length, ans_length,
                dec_state=None):

        tgt = tgt[:-1]  # exclude last target from inputs

        word_mem_bank, sent_mem_bank, quesinit, static_attn = self.encoder(
            src, ques, ans, sent_lengths, word_lengths, ques_length, ans_length)

        enc_state = self.decoder.init_decoder_state(quesinit)
        # update inputfeed by using question last embedding
        enc_state.update_state(enc_state.hidden, enc_state.hidden[0][-1].unsqueeze(0), enc_state.coverage)

        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, word_mem_bank, sent_mem_bank,
                         enc_state if dec_state is None else dec_state,
                         word_lengths, sent_lengths, static_attn)

        return decoder_outputs, attns, dec_state
