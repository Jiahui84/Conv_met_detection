## Data preparation

# This script will output 4 files altogether:

# 1.Corpus_identification_con_met.csv: Modified conventional metaphor corpus from deliberate metaphor corpus.

# 2.sentence_overview.csv: Showing the distribution the sentence by random selection (percentage of train set and test set; examples; 10-fold).

# 3.sen4prompting.csv: sentences sorted in 10 groups used for labelling.

# 4.example4prompting.csv: example sentences and their word list with labels.


# Load libraries
import pandas as pd
import pyreadstat
import numpy as np # type: ignore
import re
import random


## Covert to conventional metaphor corpus

# The conventional metaphor corpus will be adapted from the deliberate metaphor corpus, which is a spss file in sav format. The processing will include steps:

# 1.Make sure every word pair is connected with "+" and there is no space between words. For example, "I find+out the L-shaped desk". So in the later stage ChatGPT can just tokenize sentence by space (same tokenization way as that in the corpus).

# 2.The corpus consists of academic articles, converation texts, novels and News texts. Filter out others but News texts.

# 3.Reset the labels, non metaphor:0, conventional metaphor (non-deliberate metaphor):1, deliberate metaphor (activated conventional metaphor and novel metaphor, the original label):2.

# 4.Conbine words into sentence again.

# 5.Create sentence is based on docid.


# Make sure machine will always tokenize the text in the way we want
def join_multiple_words(word):
    # Ensure that word is a string
    word = str(word)
    # Use regular expressions to remove spaces around plus or minus signs
    word = re.sub(r"\s*\+\s*", "+", word)
    word = re.sub(r"\s*-\s*", "-", word)
    words = word.split()
    if len(words) > 1:
        return '+'.join(words)
    return word

# Read spss format deliberate metaphor corpus
df, meta = pyreadstat.read_sav('../data/corpus/Corpus_identification_del_met.sav')

# Filter the labelling of news texts 
df = df[df['reg'] == 'news']

# Replace the label of deliberate metaphor in DELMET column from 1to 2
df['DELMET'] = df['DELMET'].replace(1, 2)

# Find out the conventional metaphors(non-deliberate metaphors)
# When the 'type' column is 'met' and the 'DELMET' column is not 2, fill the cell corresponding to the 'DELMET' column with 1
df.loc[(df['type'] == 'met') & (df['DELMET'] != 2), 'DELMET'] = 1

# Find out the non metaphors (non-deliberate metaphors)
# When the 'type' column is not 'met', fill the cell corresponding to the 'DELMET' column with 1
df.loc[(df['type'] != 'met') & (df['DELMET'].isna()), 'DELMET'] = 0

# add two columns at the end 'context' and 'sentence_id'
df['context'] = None
df['sentence_id'] = None

# Used to save the current sentence
sentence_words = []
# Used to save the last docid of the current document
last_docid = None
# used to track the sentence order number of the current document
sentence_number = 1

# For loop DataFrame
for index, row in df.iterrows():
    # Apply new function to process words of 'word' column
    processed_word = join_multiple_words(row['word'])
    df.at[index, 'word'] = processed_word  # Make sure to update the DataFrame
    
    # Check if it is the firt word of a new document or the beginning of a new sentence
    if last_docid != row['docid'] or row['windex'] == 1:
        # If it is a new document, reset the sentence number
        if last_docid != row['docid']:
            sentence_number = 1
        # Combine the words from the previous sentence into a sentence and populate the context column
        if sentence_words:
            sentence = ' '.join(sentence_words)
            start_index = index - len(sentence_words)
            df.loc[start_index:index-1, 'context'] = sentence
            df.loc[start_index:index-1, 'sentence_id'] = f"{last_docid}_{sentence_number}"
            sentence_number += 1
            sentence_words = []  # Reset the word list of the current sentence
        last_docid = row['docid']
    
    # Add processed words to current sentence list
    sentence_words.append(processed_word)

# Populate the context column for the last sentence
if sentence_words:
    sentence = ' '.join(sentence_words)
    start_index = index - len(sentence_words) + 1
    df.loc[start_index:index, 'context'] = sentence
    df.loc[start_index:index, 'sentence_id'] = f"{last_docid}_{sentence_number}"

# Save the csv file
df.to_csv('../data/corpus/Corpus_identification_con_met.csv', index=False)
# Make sure machine will always tokenize the


## Data split

# This step is about how to distribute the corpus (e.g. percentage of train set and test set; which sentences are used as examples in the prompt for maodel to learn):

# 1.Create a csv with unique sentences.

# 2.Add a column containing labels "train" or "test", and the percentage is train 10% vs test 90%.Randomly give sentences "train" or "test" labels.

# 3.Randomly select 10 examples respectively in train set and test set. Examples must not caontain deliberate metaphors.

# 4.Randomly seplit the train set into 10 groups (10-fold).


# Make sure machine will always tokenize the text in the way we want
def join_multiple_words(word):
    # Ensure that word is a string
    word = str(word)
    # Use regular expressions to remove spaces around plus or minus signs
    word = re.sub(r"\s*\+\s*", "+", word)
    word = re.sub(r"\s*-\s*", "-", word)
    words = word.split()
    if len(words) > 1:
        return '+'.join(words)
    return word

# Read spss format deliberate metaphor corpus
df, meta = pyreadstat.read_sav('../data/corpus/Corpus_identification_del_met.sav')

# Filter the labelling of news texts
df = df[df['reg'] == 'news']

# Replace the label of deliberate metaphor in DELMET column from 1 to 2
df['DELMET'] = df['DELMET'].replace(1, 2)

# Find out the conventional metaphors (non-deliberate metaphors)
# When the 'type' column is 'met' and the 'DELMET' column is not 2, fill the cell corresponding to the 'DELMET' column with 1
df.loc[(df['type'] == 'met') & (df['DELMET'] != 2), 'DELMET'] = 1

# Find out the non-metaphors (non-deliberate metaphors)
# When the 'type' column is not 'met', fill the cell corresponding to the 'DELMET' column with 1
df.loc[(df['type'] != 'met') & (df['DELMET'].isna()), 'DELMET'] = 0

# Add two columns at the end 'context' and 'sentence_id'
df['context'] = None
df['sentence_id'] = None

# Used to save the current sentence
sentence_words = []
# Used to save the last docid of the current document
last_docid = None
# Used to track the sentence order number of the current document
sentence_number = 1

# For loop DataFrame
for index, row in df.iterrows():
    # Apply new function to process words of 'word' column
    processed_word = join_multiple_words(row['word'])
    df.at[index, 'word'] = processed_word  # Make sure to update the DataFrame
    
    # Check if it is the first word of a new document or the beginning of a new sentence
    if last_docid != row['docid'] or row['windex'] == 1:
        # If it is a new document, reset the sentence number
        if last_docid != row['docid']:
            sentence_number = 1
        # Combine the words from the previous sentence into a sentence and populate the context column
        if sentence_words:
            sentence = ' '.join(sentence_words)
            start_index = index - len(sentence_words)
            df.loc[start_index:index-1, 'context'] = sentence
            df.loc[start_index:index-1, 'sentence_id'] = f"{last_docid}_{sentence_number}"
            sentence_number += 1
            sentence_words = []  # Reset the word list of the current sentence
        last_docid = row['docid']
    
    # Add processed words to current sentence list
    sentence_words.append(processed_word)

# Populate the context column for the last sentence
if sentence_words:
    sentence = ' '.join(sentence_words)
    start_index = index - len(sentence_words) + 1
    df.loc[start_index:index, 'context'] = sentence
    df.loc[start_index:index, 'sentence_id'] = f"{last_docid}_{sentence_number}"

# Save the csv file
df.to_csv('../data/corpus/Corpus_identification_con_met.csv', index=False)

# Load csv file
df = pd.read_csv('../data/corpus/Corpus_identification_con_met.csv')

# Select columns "sentence_id" and "context"
df = df[['sentence_id', 'context', 'DELMET']]

# Label the sentences containing deliberate metaphor(s)
df['DEL'] = df.groupby(['sentence_id', 'context'])['DELMET'].transform(lambda x: 2 in x.values)

# Remove duplicate sentences
df = df.drop_duplicates(subset=['context'])

# Delete Column DEL
df = df.drop(columns=['DELMET'])

# Set a random number seed to ensure replication
np.random.seed(1)

# Add a "split" column to randomly assign sentences to train and test
df['split'] = np.where(np.random.rand(len(df)) <= 0.1, 'train', 'test')

# Add an "examples" column and initialize all values to None
df['examples'] = None

# For train set and test set, 10 sentences are randomly respectively selected and marked as example, and the sentences marked as False in the DEL column are filtered out.
train_examples = df[(df['split'] == 'train') & (df['DEL'] == False)].sample(16, random_state=1).index
test_examples = df[(df['split'] == 'test') & (df['DEL'] == False)].sample(16, random_state=1).index
df.loc[train_examples, 'examples'] = 'example'
df.loc[test_examples, 'examples'] = 'example'

# df is the DataFrame
df['10_fold'] = None

# Randomly shuffle data
train_sentences = df[(df['split'] == 'train') & (df['examples'].isnull())].sample(frac=1, random_state=1)

# Randomly split the data set into 10 folds
folds = np.array_split(train_sentences.index, 10)
for fold_number, indices in enumerate(folds, 1):
    df.loc[indices, '10_fold'] = fold_number

# Save the csv file
df.to_csv('../data/corpus/sentence_overview.csv', index=False)


## Sorting sentence for input

# Prepare a sentences in the way that can easily for loop in API: 10 groups of sentences in 10 cells in csv. In each cell, sentences are organized in the format of:

# 1.Sentence 1

# 2.Sentence 2

# ......

# 18.Sentence 18


# Set random seed
random.seed(42)  # Specify your desired seed value in the parentheses

# Read CSV file into a DataFrame
file_path = '../data/corpus/sentence_overview.csv'
df = pd.read_csv(file_path)

# Create a function to add sentence numbers
def number_sentences(group):
    sentences = []
    for s, sid in zip(group['context'].tolist(), group['sentence_id'].tolist()):
        sentences.append(f"{s} (sentence_id: {sid})")
    return "\n".join(sentences)

# Group by '10_fold' and apply the numbering function
grouped_df = df.groupby('10_fold').apply(number_sentences).reset_index(name='query_sentence')

# Add the 'sentence_id' column to the DataFrame
sentence_ids = df.groupby('10_fold').apply(lambda x: "\n".join(f"{sid}" for sid in x['sentence_id'].tolist())).tolist()
grouped_df['sentence_id'] = sentence_ids

# Save the results to a new CSV file
grouped_df.to_csv('../data/corpus/sen4prompting.csv', index=False)


## Sorting eamples for N-shot prompting

# Examples require sentence and corresponding word labelling.This step is about how to get them and store them in a csv.


# Load existing data
sentence_overview_df = pd.read_csv('../data/corpus/sentence_overview.csv')
corpus_df = pd.read_csv('../data/corpus/Corpus_identification_con_met.csv')

# Filter out rows labelled with "example" in "examples" column
filtered_sentence_overview_df = sentence_overview_df[sentence_overview_df['examples'] == 'example'][['examples', 'split', 'context', 'sentence_id']]

# Function to create a word list
def create_word_list(context):
    relevant_rows = corpus_df[corpus_df['context'] == context]
    word_list = [f"{word}{':1' if delmet else ''}" for word, delmet in zip(relevant_rows['word'], relevant_rows['DELMET'])]
    return "\n".join(word_list)

filtered_sentence_overview_df['word_list'] = filtered_sentence_overview_df['context'].apply(create_word_list)

# Add a new grouping column
def assign_groups(df, split, group_sizes):
    split_df = df[df['split'] == split]
    for size in group_sizes:
        selected_indices = split_df.sample(n=size if len(split_df) >= size else len(split_df)).index
        df.loc[selected_indices, 'group'] = size
        split_df = split_df.drop(selected_indices)

# Separate groups for 'train' and 'test'
assign_groups(filtered_sentence_overview_df, 'train', [1, 5, 10])
assign_groups(filtered_sentence_overview_df, 'test', [1, 5, 10])

# Save as a new csv file
filtered_sentence_overview_df.to_csv('../data/corpus/example4prompting.csv', index=False)
