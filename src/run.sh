#!/bin/bash

MEM=5000

if [[ $1 == 'att' ]]; then
	echo 'nmt_dynet_attention'
	python nmt_dynet_attention.py \
			../data/train.src.new \
			../data/train.tgt.new \
			../data/dev.src \
			../data/dev.tgt \
			../data/final_nmt_dynet_attention \
			--dynet-mem $MEM
elif [[ $1 == 'beam' ]]; then
	echo 'beam search'
	python nmt_beam.py \
		--dynet-mem $MEM
else
	echo 'nmt_dynet'
	python nmt_dynet.py \
		../data/train.src \
		../data/train.tgt \
		../data/dev.src \
		../data/dev.tgt \
		../data/final_nmt_dynet \
		--dynet-mem $MEM
fi
