#!/bin/bash

PROJ=/path/to/Distractor-Generation-RACE

export CUDA_VISIBLE_DEVICES=$1
MODEL=$2
DATE=$3

python -u train_single.py \
        -word_vec_size=300 \
        -share_embeddings \
        -rnn_size=600 \
        -word_encoder_layers=2 \
        -sent_encoder_layers=1 \
        -question_init_layers=2 \
        -lambda_question=0.5 \
        -lambda_answer=-1 \
        -data=data/processed \
        -save_model=data/model/${DATE}_${MODEL} \
        -save_checkpoint_steps=3000 \
        -gpuid=0 \
        -pre_word_vecs_enc=data/processed.glove.enc.pt \
        -pre_word_vecs_dec=data/processed.glove.dec.pt \
        -batch_size=32 \
        -valid_steps=3000 \
        -valid_batch_size=16 \
        -train_steps=45000 \
        -optim=adagrad \
        -adagrad_accumulator_init=0.1 \
        -learning_rate=0.1 \
        -learning_rate_decay=0.5 \
        -start_decay_steps=30001 \
        -decay_steps=3000 \
        -seed=1995 \
        -report_every=300

