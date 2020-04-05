""" Hierarchal Decoder"""

from __future__ import division
import torch
import torch.nn as nn

from onmt.utils.misc import aeq
from onmt.utils.rnn_factory import rnn_factory
from onmt.distractor.attention import HierarchicalAttention
from onmt.utils.logging import logger

class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input_feed, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input_feed, (h_0[i], c_0[i]))
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input_feed, (h_1, c_1)


class StackedGRU(nn.Module):
    """
    Our own implementation of stacked GRU.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input_feed, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input_feed, hidden[0][i])
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        return input_feed, (h_1,)

class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()

class RNNDecoderState(DecoderState):
    """ Base class for RNN decoder state """

    def __init__(self, hidden_size, rnnstate):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = self.hidden[0].size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = self.hidden[0].data.new(*h_size).zero_() \
                              .unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        """ Update decoder state """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [e.data.repeat(1, beam_size, 1)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]

    def map_batch_fn(self, fn):
        self.hidden = tuple(map(lambda x: fn(x, 1), self.hidden))
        self.input_feed = fn(self.input_feed, 1)

class HierDecoder(nn.Module):
    """
    Hierarchal Decoder for sent and word level
    Args:
        Unfinished!!! focus on encoder first~
    """
    def __init__(self, rnn_type,
                 bidirectional_encoder,
                 num_layers,
                 hidden_size,
                 attn_type="general",
                 dropout=0.0, embeddings=None):
        super(HierDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        self.attn = HierarchicalAttention(hidden_size, attn_type=attn_type)


    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size

    def forward(self, tgt, word_memory_bank, sent_memory_bank, state,
                word_memory_lengths, sent_memory_lengths, static_attn,
                step=None):
        # Check
        assert isinstance(state, RNNDecoderState)
        # tgt.size() returns tgt length and batch
        tgt = tgt.unsqueeze(-1)
        _, tgt_batch, _ = tgt.size()
        _, sent_memory_batch, _ = sent_memory_bank.size()
        aeq(tgt_batch, sent_memory_batch)
        # END

        # Run the forward pass of the RNN.
        decoder_final, decoder_outputs, attns = self._run_forward_pass(
            tgt, word_memory_bank, sent_memory_bank, state,
            word_memory_lengths, sent_memory_lengths,
            static_attn)

        # Update the state with the result.
        final_output = decoder_outputs[-1]
        coverage = None
        state.update_state(decoder_final, final_output.unsqueeze(0), coverage)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: decoder_outputs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(decoder_outputs) == list:
            decoder_outputs = torch.stack(decoder_outputs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])

        return decoder_outputs, state, attns

    def init_decoder_state(self, encoder_final):
        """ Init decoder state with last state of the encoder """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size,
                                   tuple([_fix_enc_hidden(enc_hid)
                                          for enc_hid in encoder_final]))
        else:  # GRU
            return RNNDecoderState(self.hidden_size,
                                   _fix_enc_hidden(encoder_final))

    def _run_forward_pass(self, tgt, word_memory_bank, sent_memory_bank,
                          state, word_memory_lengths, sent_memory_lengths,
                          static_attn):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden

        # Input feed concatenates hidden state with
        # input at every time step.
        for outidx, emb_t in enumerate(emb.split(1)):
            # logger.info('generate %d word' %outidx)
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)

            rnn_output, hidden = self.rnn(decoder_input, hidden)

            # attn
            decoder_output, attn = self.attn(
                rnn_output,
                word_memory_bank,
                word_memory_lengths,
                sent_memory_bank,
                sent_memory_lengths,
                static_attn)

            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            decoder_outputs += [decoder_output]
            attns["std"] += [attn]

        # Return result.
        return hidden, decoder_outputs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = StackedLSTM
        else:
            stacked_cell = StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

