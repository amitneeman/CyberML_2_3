# Early Sexual Predator Detection Lab

With this code you can train and evaluate language models to create early sexual predator detection (eSPD) systems that aim to detect grooming in chats.
The repository provides an experimental setup for this which includes training and evaluation. Specifically, you can fine-tune BERT language models using the libraries
- [Flair](https://github.com/flairNLP/flair) for BERT<sub>Large</sub> and
- [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification) for BERT<sub>Base</sub> and MobileBERT. These models also work on mobile devices.

This is part of a larger project on [early sexual predator detection](https://early-sexual-predator-detection.gitlab.io/) which includes a paper [1]. You can also reproduce the results from this paper. For this, you need to obtain the [eSPD datasets](https://gitlab.com/early-sexual-predator-detection/eSPD-datasets) we used.

![Warning. This code is for research purposes only. Trained models will not be able to detect real grooming attempts. Do not use such models in practice. For more info, read the paper.](do_not_use.svg)
## Setup

Make sure you have `Python>=3.6.9` and the latest versions of `flair` and `tflite-model-maker` . You also need `tqdm` for progress bars and `numpy`. To get these packages, run
```
pip install flair tflite-model-maker tqdm numpy
```
in your favourite virtual environment. The code in this repo was tested with `tflite-model-maker==0.3.2` and `flair==0.8.0.post1` and may not work with newer versions.


##  Training models


Use
- `train.py` to fine-tune BERT<sub>Base</sub> or MobileBERT with [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification) or
- `train_flair.py` to fine-tune BERT<sub>Large</sub> with [Flair](https://github.com/flairNLP/flair).

## Evaluating models

There are multiple ways of evaluating models:

- `evaluate_f1.py`
    - simply evaluates the F1 of a language model for classifying segments in a dataset
- `evaluate_f1_over_percentage_of_information.py`
    - evaluates F1 for percentages of characters of segments (see Section&nbsp;5.2. in&nbsp;[1]). We do not recommend this type of evaluation, but you can use it to compare the language model to other works that used the dataset `VTPAN` [2]
    - with this you can generate the plot from Figure&nbsp;5 in&nbsp;[1] using the R file `mean_multiple_f1_over_percentage_of_information.R`
- `annotate_datapack_with_predictions.py`
    - for each chat in a datapack, this annotates each message with the prediction of a language model for the chat up to the message
    - **This first step is necessary for all further evaluations below**
    - `eval_mode` can be set to `segments` or `full` depending on whether you want to annotate segments or complete predator chats.
    - in normal mode, this evaluation is very slow because a chat with n messages has to be classified n times
    - We recommend using the `segments_fast` and `full_fast` modes instead, which stop annotating the chat as soon as a warning for a message is raised. Use this if you are OK with partial annotations e.g. when you dont want to manually inspect the predictions for all messages in a chat with [chat-visualizer](https://gitlab.com/early-sexual-predator-detection/chat-visualizer).
- `message_based_evaluation.py`
    - evaluates warning precision, recall, F1 and latency for an eSPD system on PANC, i.e. for *segments* of chats (see Section&nbsp;5.1 in&nbsp;[1])
    - this is done for all skepticism values s=1,…,10
    - this needs annotated datapacks with `eval_mode` `segments`
    - with this you can generate the results for
        - Table&nbsp;2 and the plots for
        - Figures&nbsp;3&nbsp;and&nbsp;4 in&nbsp;[1] using the R files `mean_warning_latency_distribution.R` and `mean_metrics_by_skepticism.R` respectively
- `full_length_chats_evaluation.py`
    - evaluates warning latencies for *full-length predator chats* for an eSPD system and a given skepticism
    - these latencies are also evaluated with `message_based_evaluation.py` but this script allows evaluating them in isolation
    - this script needs annotated datapacks with `eval_mode` of `full` or `full_fast`
    - with this you can generate results and plots for Figure&nbsp;3 in&nbsp;[1] using the R file `mean_warning_latency_distribution.R`

# Plots

To generate the original plots from&nbsp;[1], you need [our evaluation results](https://mega.nz/folder/SZMnDQaY#E5NXBL8uaYAhc5ebZEeuEg). Extract the `resources/` folder from the evaluation results archive in this directory. Then use the `.R` files in `r_scripts/*.R`.

You can open the scripts in RStudio to generate the plots.

You can also run
```
R < r_scripts/some_script.R --no-save
```
from the command line, which saves an Rplots.pdf file to the project directory.

## Folders

### datasets/`DATASET_INDICATOR`/

Contains the train and test splits for the datasets you want to train and test your models on. To populate this, use the [sexual-predator-datasets](https://gitlab.com/early-sexual-predator-detection/eSPD-datasets) repository.


### resources/`run`/

Contains model files and evaluation results for a model with run ID `run`. Run IDs are used to identify different runs and look like `2020-11-22_00-45-54__bert_classifier_on_PANC_with_seq-len-512` for example.

## Contributions

You are welcome to contribute :)

## References

[1] Vogt, Matthias, Ulf Leser, and Alan Akbik. "[Early Detection of Sexual Predators in Chats](https://aclanthology.org/2021.acl-long.386/)." Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 2021.

[2] Pastor Ĺopez-Monroy, A., Gonźalez, F. A., Montes-Y-Ǵomez, M., Escalante, H. J. & Solorio, T. Early text classification using multi-resolution concept representations. in NAACL HLT 2018 - 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies - Proceedings of the Conference (2018). doi:10.18653/v1/n18-1110
