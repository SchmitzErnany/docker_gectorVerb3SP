# gectorptbr

This repository contains all the tools needed for us to:

* synthetically introduce errors in the original data. This is done by _PIE/errorify/ptbr/error.py_;
* fine-tune the BERTimbau model in order to have a model which predicts the correct grammatical replacement when a writing error takes place. The training is done by _./train.py_.

In the folder _google_colab/_, you will find the jupyter notebook that contains all the steps from downloading the wiki-sentences.txt to the inference of the fine-tuned model. We assume that you will work in Google Colab and that this repository will be inside Google Drive, since many paths inside the notebook take into account this repository being in Google Drive.

**What is missing** in this repository is only the **wiki-sentences.txt** which has the size of ~780mb. This file you will have to put inside the *preprocessable_files/* folder so that the jupyter notebook can call it without warning.

