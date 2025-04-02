# Introduction
This project presents a workflow to explore if GPT4 can detect conventional metaphors with prompting approaches. For the Hypothesis, we assume that model performance on this task will improve with the increase of examples provided to the model, until reaching the model's capability limit. And more examples can teach the model more patterns to distinguish conventional metaphors from others. Thus, we design a two-stage experiment to verify this:

1.	Zero-shot (bare task description to see what the model does “out of the box”)
2.	N(1, 5, 10)-shot prompting (providing N examples)
-	providing sentences with word-level labels 


# Structure of the documents
There are 6 folders:

## background check
It includes 6 documents of the chats with GPT4 API and ChatGPT4 about the background information of metaphors/conventional metaphors

## data
1.	corpus
Corpus_identification_del_met.csv: original deliberate metaphor corpus with deliberate metaphors labelled with 1
Corpus_identification_con_met.csv: adapted conventional metaphor corpus with conventional metaphor labelled with 1, non-conventional metaphors labelled with 2 and non-metaphors labelled with 0
secret_held_up_with_sentence.csv: secret held out dataset labelled by group members as a comparison for the corpus data in model performance
sentence_overview.csv: overview of sentence split of trainset, testset and example set
example4prompting.csv: extract example sentences and word list with labels
sen4prompting.csv: extract input sentences for model labelling

2.	prompt set
Prompt_map.docx: Explain the catagorization of the prompts (prompt features)
25_prompts_0shot.csv: prompts for 0-shot prompting
25_prompts_nshot.csv: prompts for n-shot prompting

## codes
Preprocess_corpus.ipynb: Pre-processing corpus for model labelling and prompts premutation for model use
Running_through_API.ipynb: Run sentences for labelling through the models with different prompts, and post-process the outputs for evaluation
Get_classification_report.py: get performance report of Accuracy, F1, Recall, Precision for model outputs
wordcat_eval.ipynb: Extract True Positives (TP), True Negatives (TN) and False Positives (FP) words and their wordcat. Conduct statistical analysis of the wordcat
McNemar test.ipynb: Evaluate model's stability from two directions ——  comparison of same prompt in different shot prompting and comparison of different prompts in the same shot prompting

## outputs
output_0: raw outputs, sorted outputs and results of 0-shot prompting stage
output_1: raw outputs, sorted outputs and results of 1-shot prompting stage
output_5: raw outputs, sorted outputs and results of 5-shot prompting stage
output_10: raw outputs, sorted outputs and results of 10-shot prompting stage
secret: repeat the 0-10 shot prompting experiments above
+words.txt: summarize all the word pairs in the corpus
Output_Sorting_Guidelines.docx: guidelines for sorting out the raw model outputs

## analysis
1.	wordcat error analysis
raw: raw TP, FP, FN extract from the output sorted files
stats: calculate the wordcats of TP, FP, FN in three times experiments
average_stats: average stats of three times experiment and visualize them in figures

2.	McNemar test
Comparison_across_shots: see whether there is significant difference for same prompt in different shot prompting stages
Comparison_between_prompts_in_different_shots: see whether there is significant difference for different prompt in the same shot prompting stage

## test
0-shot_keyword: results for testing same prompts with different key words: lexicalized metaphor, metaphors, conventional metaphors but only NVAJ, conventional metaphors but only NVAJ, metaphors but only NVAJ
IBO_labeling: results for testing using IBO labeling schema in the prompt
trial_50: results for providing 50 examples as 50-shot prompting
AutoPrompt-main: results of prompts revised by the prompt improvement tool —— AutoPrompt

#Replication
If you want to get the performance results yourself, you can run through the scripts: Step3_Get_classification_report.py, Last_Step_wordcat_eval.ipynb, Last_Step_McNemar test.ipynb directly
Run Step3_Get_classification_report.py with this line: 
(output_num, num from 0/1/5/10)

Output_Sorting_Guidelines.docx: Explains the problems encountered while sorting out model outputs for evaluation and corresponding solutions
python Get_classification_report.py "../outputs/output_0/csv/sorted/(conventional)*_output_sorted.csv" conventional_results.log

raw: the original model outputs of different variation of prompts and different prompts in txt file

sorted: roughly sort the raw output in clear format (word or word:label, one word or word:label per row)

csv: 

- raw: conbine manual annotation word:label and model output word:label in csv

- sorted: sort out the model outputs especially the tokenization problems so they can align to the manual word list

- csv: 

	class_0: classification results on non conventional metaphors  of three-time experiment, and average results of the three-time experiment

	class_1: classification results on conventional metaphors of  three-time experiment, and average results of the three-time experiment

	general: evaluation results for every variation of prompt and every prompt for three-time experiment, and average results of the three-time experiment

- results_log: log files of code running records of the performance results


If you want to replicate the whole experiment:

Open terminal -> locate to code folder in the main folder

Step 1: Preprocess raw corpus to generate files for input in later stages: Python Step1_Preprocess_corpus.py

Step 2: Running through API to get raw outputs: Python Step2_Running_through_API.py 
Note: 1. pip install openai==0.28.0
           2. Please enter your OpenAI API key: 
           3. Please select the experiment run (first, second, third):  
           4. Please select the shot number (0, 1, 5, 10) or press Enter to run all: 

Step 3: You need to manual check all the outputs in "outputs/output_i/sorted/x_time_output/" and do the rough sorting into consistent format: word list of one word/ one word : 1 per row and saved the checked version, and then run the command line below to turn them into csv files: Python Step3_generate_csv_for_calculation.py

Step 4: You need to manually check and make sure all the outputs/words and labels of different prompts align to the ground truth words and labels in format, and then run the command line to get classification report: Python Step4_Get_classification_report.py

Step 5: Run the command lines separately to get evaluation results of word category error analysis and McNemar test: Python Last_Step_wordcat_eval.py ; Python Last_Step_McNemar_test.py

Feel free to just run specific steps. It's worth noting that for the model, same input can have different output, so if you run Python Step2_Running_through_API.py, it's possible you will get outputs different from our project.
