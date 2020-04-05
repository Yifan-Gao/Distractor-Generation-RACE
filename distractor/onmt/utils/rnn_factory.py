"""
 RNN tools
"""
from __future__ import division

import torch.nn as nn


def rnn_factory(rnn_type, **kwargs):
    """ rnn factory, Use pytorch version when available. """
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        raise NotImplementedError
    else:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq
