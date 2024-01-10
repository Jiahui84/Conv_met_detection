## Introduction
This project presents a robust workflow for experimenting with large language models (LLM) for conventional metaphor detection, such as ChatGPT. Our methodology involves permutating prompts, applying 10-fold cross-validation, and analyzing the outcomes to measure confidence intervals and real differences in model performance.

## Getting Started
To utilize this workflow, clone the repository and follow the setup instructions provided in the documentation. The workflow is designed to allow easy permutation of prompts (P) and variable k-fold settings to best suit testing needs.

## Usage
Once set up, the workflow can be used to sample from a binomial distribution of prompts, train the LLM with selected prompts, and perform a thorough analysis of the results. This is essential for anyone looking to fine-tune LLMs or compare prompting strategies effectively.

## Replication
You can replicate the whole experiment by just running through the scripts in the folder "code".
If you just want to get the evalutaion results, you can just run the last cells in "Running_through_API.ipynb" in folder code.
The whole prompt set and conventioanl metaphor corpus are in "data folder".

## Structure of the folder

There are altogether three folders:

## codes
Preprocess_corpus.ipynb: Pre-processing corpus for model labelling and prompts premutation for model use
Running_through_API.ipynb: Run sentences for labelling through the models with different prompts, and post-process the outputs for evaluation

## data

#corpus

Corpus_identification_del_met.sav: original deliberate corpus, and the conventional metaphor corpus will be an adapted version from it

Corpus_identification_con_met.csv: csv version deliberate metaphor corpus with sentence-level context for each word or word pairs

example4prompting.csv: sentences randomly selected for 1, 5, 10 shot learning

expanded_secret_held_up_with_sentence.csv: manual annotated corpus to test if model performance on conventional metaphor detection is influenced by the existing public corpora (deliberate metaphor corpus, VUAMC corpus)

sen4prompting.csv: sentences in random order split in 10 groups for model labelling 

sentence_overview.csv: showing which sentences are for the 10% train data, and 90% test data, and among the train data and test data, how many sentences are used as examples

#prompt_set

25_Prompts_Cov_met_sen.csv: Prompts for model labelling with slight variations in expression. All the prompts are co-created with ChatGPT interface

Prompt_map.docx: Explain the catagorization of the prompts (prompt features)

selected_permuted_prompts.csv: Output of prompts after permutation


##outputs

Output_Sorting_Guidelines.docx: Explains the problems encountered while sorting out model outputs for evaluation and sorresponding solutions

#output_0

raw: the original model outputs of different variation of prompts and different prompts in txt file

sorted: roughly sort the raw output in clear format (word or word:label, one word or word:label per row)

csv: 

- raw: conbine manual annotation word:label and model output word:label in csv

- sorted: sort out the model outputs especially the tokenization problems so they can align to the manual word list

- csv: evalutaion results for every variation of prompt and every prompt for three-time experiment, and average results of the three-time experiment

More folders will be created for 1-shot, 5-shot and 10-shot (output_1, output_5, output_10)