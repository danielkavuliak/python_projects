#!/bin/bash

#autor: Michal Kabac(xkabac00)
#predmet: SUR
#rok: 2021/2022

#txt-for-face - cesta ku txt s vysledkami pre obrazky
#txt-for-audio - cesta ku txt s vysledkami pre rec
#save-results-dir - miesto kde sa ma ulozit vysledne txt + jeho nazov

#do premennej BASE ide cesta ku projektu
BASE=/Volumes/XKABAC00/SUR_projekt2021-2022

python3 -W ignore $BASE/src/combine_results.py \
    --txt-for-face=$BASE/RF_image.txt \
    --txt-for-audio=$BASE/GMM_audio.txt \
    --save-results-dir=$BASE/model_combination2.txt \
