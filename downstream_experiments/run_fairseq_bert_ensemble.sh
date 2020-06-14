#!/usr/bin/env bash

BERT_ENTITY_CHECKPOINT=$1

cd "fairseq/"

python train.py data-bin/bertencoder--wmt14_en_de --optimizer adam --lr 0.00005 --clip-norm 0.1 --dropout 0.2 --max-tokens 1000 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr-scheduler fixed --force-anneal 200 --arch bert_transformer_iwslt_en_de --save-dir checkpoints/bert_transformer_wmt14_en_de_ft_20 --fp16 --finetuning 2 --save-interval-updates 10000  --validate-interval 10000

python train.py data-bin/bertencoder--wmt14_en_de --optimizer adam --lr 0.00005 --clip-norm 0.1 --dropout 0.2 --max-tokens 1000 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr-scheduler fixed --force-anneal 200 --arch bert_transformer_iwslt_en_de --save-dir checkpoints/bert_transformer_wmt14_en_de_ft_10-entity --fp16 --finetuning 1 --save-interval-updates 10000  --validate-interval 10000 --load_bert_checkpoint $BERT_ENTITY_CHECKPOINT

python setup.py build develop

python generate.py data-bin/bertencoder--wmt14_en_de --beam 5 --remove-bpe --batch-size 128 --path
  checkpoints/bert_transformer_wmt14_en_de_ft_10/checkpoint_best.pt | tee checkpoints/bert_transformer_wmt14_en_de_ft_10/gen.out

python generate.py data-bin/bertencoder--wmt14_en_de --beam 5 --remove-bpe --batch-size 128 --path checkpoints/bert_transformer_wmt14_en_de_ft_10-entity/checkpoint_best.pt | tee checkpoints/bert_transformer_wmt14_en_de_ft_10-entity/gen.out