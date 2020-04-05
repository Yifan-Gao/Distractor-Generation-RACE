#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import argparse
import codecs
import os
import math
from functools import reduce

from tqdm import tqdm

import torch
import torchtext

from itertools import count
from onmt.utils.misc import tile

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.opts as opts


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model, model_opt = \
        onmt.model_builder.load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)

    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam", "report_bleu",
                        "data_type", "replace_unk", "gpu", "verbose", "fast",
                        ]}

    translator = Translator(model, fields, global_scorer=scorer,
                            out_file=out_file, gold_file=opt.target,
                            report_score=report_score,
                            copy_attn=model_opt.copy_attn, logger=logger,
                            **kwargs)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 model,
                 fields,
                 beam_size,
                 n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 logger=None,
                 gpu=False,
                 dump_beam="",
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[],
                 use_filter_pred=False,
                 data_type=None,
                 replace_unk=False,
                 report_score=True,
                 report_bleu=False,
                 report_rouge=False,
                 verbose=False,
                 out_file=None,
                 gold_file=None,
                 fast=False):
        self.logger = logger
        self.gpu = gpu
        self.cuda = gpu > -1

        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)
        self.use_filter_pred = use_filter_pred
        self.replace_unk = replace_unk
        self.data_type = data_type
        self.verbose = verbose
        self.out_file = out_file
        self.gold_file = gold_file
        self.report_score = report_score
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.fast = fast

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate(self,
                  data_path=None,
                  data_iter=None,
                  batch_size=None):
        """
        Translate content of `data_iter` (if not None) or `data_path`
        and get gold scores.

        Note: batch_size must not be None
        Note: one of ('data_path', 'data_iter') must not be None

        Args:
            data_path (str): filepath of source data
            data_iter (iterator): an interator generating source data
                e.g. it may be a list or an openned file
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        assert data_iter is not None or data_path is not None

        if batch_size is None:
            raise ValueError("batch_size must be set")
        data = inputters. \
            build_dataset(self.fields,
                          data_type=self.data_type,
                          data_iter=data_iter,
                          data_path=data_path,
                          use_filter_pred=self.use_filter_pred)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        def sort_key(ex):
            """ Sort using length of source sentences. """
            return len(ex.src)
        # data.fields = self.fields
        data_iter = torchtext.data.Iterator(dataset=data,
                                            batch_size=batch_size,
                                            device=cur_device,
                                            train=False, sort=False,
                                            sort_key=sort_key,
                                            repeat=False,
                                            sort_within_batch=False,
                                            shuffle=False)

        builder = onmt.translate.TranslationBuilder(
            data, self.data_type, self.fields,
            self.n_best, self.replace_unk, has_tgt=False)

        # Statistics
        translated = []
        for batch in tqdm(data_iter):
            batch_data = self.translate_batch(batch, data)
            translations = builder.from_batch(batch_data)
            translated.extend(translations)
        return translated


    def translate_batch(self, batch, data):
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        vocab = self.fields["tgt"].vocab

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[inputters.PAD_WORD],
                                    eos=vocab.stoi[inputters.EOS_WORD],
                                    bos=vocab.stoi[inputters.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a):
            return torch.tensor(a, requires_grad=False)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        src, sent_lengths, word_lengths = batch.src[0], batch.src[1], batch.src[2]
        ques, ques_length = batch.question[0], batch.question[1]
        ans, ans_length = batch.answer[0], batch.answer[1]

        word_mem_bank, sent_mem_bank, quesinit, static_attn = self.model.encoder(
            src, ques, ans, sent_lengths, word_lengths, ques_length, ans_length)

        enc_state = self.model.decoder.init_decoder_state(quesinit)
        # update inputfeed by using question last embedding
        enc_state.update_state(enc_state.hidden, enc_state.hidden[0][-1].unsqueeze(0), enc_state.coverage)

        # (2) Repeat src objects `beam_size` times.
        word_mem_bank = var(word_mem_bank.repeat(1, beam_size, 1, 1))
        sent_mem_bank = rvar(sent_mem_bank.data)
        static_attn = static_attn.repeat(beam_size, 1)
        sent_lengths = sent_lengths.repeat(beam_size)
        word_lengths = word_lengths.repeat(beam_size, 1)

        enc_state.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Run one step.
            dec_out, dec_states, attn = \
                self.model.decoder(inp, word_mem_bank, sent_mem_bank,
                                   enc_state, word_lengths,
                                   sent_lengths, static_attn)
            dec_out = dec_out.squeeze(0)
            # (b) Compute a vector of batch x beam word scores.
            out = self.model.generator.forward(dec_out).data
            out = unbottle(out)
            # beam x tgt_vocab
            beam_attn = unbottle(attn["std"])

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j], beam_attn.data[:, j, :])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        ret["batch"] = batch

        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret


