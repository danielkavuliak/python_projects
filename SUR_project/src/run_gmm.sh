#!/bin/bash

#autor: Michal Kabac(xkabac00)
#predmet: SUR
#rok: 2021/2022


#--directory-train-detect - lokacia suboru pre trenovacie data hladanej osoby
#--directory-train-not-detect - lokacia suboru pre trenovacie data ostatnych osob
#--directory-val-detect - lokacia suboru pre validacne data hladanej osoby
#--directory-val-not-detect - lokacia suboru pre validacne data ostatnych osob
#--directory-eval - lokacia suboru pre testovacie data
#--save-trained-model - ulozenie modelu
#--path-save-trained-model - cesta pre ulozenie modelu
#--save-results-dir - miesto ulozenia a nazov suboru s vysledkami
#do premennej BASE ide cesta ku projektu

BASE=/Volumes/XKABAC00/SUR_projekt2021-2022

python3 -W ignore $BASE/src/gmm.py \
    --directory-train-detect=$BASE/target_train \
    --directory-train-not-detect=$BASE/non_target_train \
    --directory-val-detect=$BASE/target_dev \
    --directory-val-not-detect=$BASE/non_target_dev \
    --directory-eval=$BASE/eval \
    --save-trained-model=true \
    --path-save-trained-model=./ \
    --save-results-dir=$BASE/GMM_audio_augumented.txt \
