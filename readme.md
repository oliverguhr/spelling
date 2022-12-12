# Spelling correction based on pretrained transformer models

## Purpose

This is an attempt to create a model that is able to fix spelling errors and 
common typos. 

An english work in progress model and interactive demo can be found [here](https://huggingface.co/oliverguhr/spelling-correction-english-base) and a german version [here](https://huggingface.co/oliverguhr/spelling-correction-german-base).

## Install

1. `python -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`

## Generate Training Data

To generate the training data simply run these two scripts:

1. `sh convert_leipzig_data.sh`
2. `python generate_dataset.py`

Optional: If you want to combine multiple languages you can edit and run the "combine.sh" script.


By default this will create the english dataset. To switch to a different language, 
you need to change the language tag in those two scripts.


## How to train a model:

For english run `sh train_bart_model.sh` or `train_de_bart_model.sh` for the german model.

## Contribute:

This is an open research project, improvements and contributions are welcome. 
If we achive promising results, we will publish them in a more formal way (paper). 
All contributers will be recognized.

## Open Questions

* How do evaluate the quality of the model, apart from using CER on syntactic data?
* What are good data sets to train on?


## Possible Datasets:

* https://github.com/snukky/wikiedits
* https://github.com/mhagiwara/github-typo-corpus
    *  Too much noise, does not work well.
