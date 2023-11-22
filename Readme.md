## Introduction
This project presents a robust workflow for experimenting with large language models (LLM) for conventional metaphor detection, such as ChatGPT. Our methodology involves permutating prompts, applying 10-fold cross-validation, and analyzing the outcomes to measure confidence intervals and real differences in model performance.

## Getting Started
To utilize this workflow, clone the repository and follow the setup instructions provided in the documentation. The workflow is designed to allow easy permutation of prompts (P) and variable k-fold settings to best suit testing needs.

## Usage
Once set up, the workflow can be used to sample from a binomial distribution of prompts, train the LLM with selected prompts, and perform a thorough analysis of the results. This is essential for anyone looking to fine-tune LLMs or compare prompting strategies effectively.

## Repliccation
You can replicate the whole experiment by just running through the scripts in the folder "code".
If you just want to get the evalutaion results, you can just run the last ? cells in "Running_through_API.ipynb" in folder code.
The whole prompt set and conventioanl metaphor corpus are in "data folder".

