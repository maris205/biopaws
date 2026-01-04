#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
lang=$1
for ((i=0;i<100;i++))
do
	echo "----------------------------------------------"$i
	python gpt2_test_protein.py $i $lang >> gpt2_test_protein_${lang}.json
done

