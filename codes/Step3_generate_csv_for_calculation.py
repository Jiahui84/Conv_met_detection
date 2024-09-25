#import needed libraries
import pandas as pd
import openai
print(openai.__version__)
import os
import re

# Function to process text files by removing and re-adding serial numbers
def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    processed_lines = []
    new_line_number = 1
    for line in lines:
        # Only remove serial numbers starting with a number and a period
        line = re.sub(r'^\d+\.\s+', '', line)

        # Check if there is a blank row
        if line.strip():
            # Add new digital serial number
            processed_lines.append(f"{new_line_number}. {line}")
            new_line_number += 1
        else:
            # Save the blank row and reset digital serial number
            processed_lines.append('\n')
            new_line_number = 1

    # Update the files
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in processed_lines:
            file.write(line)

# Process all files in the given directory
def process_all_files_in_directory(directory_path):
    # For loop every file in the folder
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            process_text_file(file_path)
            print(f"Processed {file_path}")

# Function to merge all the outputs into a single list for every prompt
def merge_files_by_prefix(directory, prefix):
    all_lines = []  # Initialize a list to store all lines from the files
    for i in range(1, 11):  # 10 files for 10-fold
        file_name = f"{prefix}_{i}.txt"
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                all_lines.extend([line.strip() for line in lines if line.strip()])
    return all_lines

# Get unique prefixes from file names in the specified directory
def get_unique_prefixes(directory):
    prefixes = set()
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            prefix = file.split('_')[0]
            prefixes.add(prefix)
    return prefixes

# Load corpus data
corpus_df = pd.read_csv('../data/corpus/Corpus_identification_con_met.csv')  # Original manual annotation corpus
prompting_df = pd.read_csv('../data/corpus/sen4prompting.csv', encoding='latin1')  # Sentences after random permutation

# Extract the "sentence_id" column into a list
query_sentences = prompting_df['sentence_id'].tolist()

# Function to clean and split sentences when multiple IDs are contained in each cell, separated by new lines
def clean_and_split_sentences(sentences):
    clean_ids = []
    for cell in sentences:
        # Split IDs on new lines within the cell
        ids = cell.split('\n')
        for id in ids:
            cleaned_id = id.strip()
            if cleaned_id:
                clean_ids.append(cleaned_id)
    return clean_ids

# Batch process directories and save CSV files for each folder
output_folders = ['output_0', 'output_1', 'output_5', 'output_10']
output_types = ['first_time_output', 'second_time_output', 'third_time_output']

# Iterate through each output folder
for output_folder in output_folders:
    # Process each output type (first, second, third time)
    for output_type in output_types:
        directory_path = f'../outputs/{output_folder}/sorted/{output_type}'
        
        # Process all text files in the directory
        process_all_files_in_directory(directory_path)
        
        # Get unique prefixes and merge texts
        sentence_ids = clean_and_split_sentences(query_sentences)
        unique_prefixes = get_unique_prefixes(directory_path)

        merged_texts = {}
        # For each prefix, call merge_files_by_prefix function and store the result in the dictionary
        for prefix in unique_prefixes:
            merged_texts[prefix] = merge_files_by_prefix(directory_path, prefix)

        # Initialize an empty DataFrame for storing matches
        matched_df = pd.DataFrame(columns=['word', 'DELMET', 'wordcat', 'context'])

        # Iterate over each sentence ID in the list and find matches in the corpus
        for sentence_id in sentence_ids:
            matches = corpus_df[corpus_df['sentence_id'] == sentence_id]
            if not matches.empty:
                # For each match, add the relevant data to the matched_df using concat
                for _, match in matches.iterrows():
                    delmet_int = int(match['DELMET'])
                    new_row = pd.DataFrame({
                        'word': [match['word']],
                        'DELMET': [delmet_int],
                        'wordcat': [match['wordcat']],
                        'context': [match['context']]
                    })
                    matched_df = pd.concat([matched_df, new_row], ignore_index=True)

        # Write matched DataFrame into csv file
        csv_output_filename = f"../outputs/{output_folder}/csv/raw/(conventional_{output_folder.split('_')[1]}){output_type}.csv"
        matched_df.to_csv(csv_output_filename, index=False)

        # Combine manual annotation and model output
        # Process every list for every prompt into columns in the csv file
        def process_list(lst, list_number):
            # Split every row into order, word, label
            parsed_data = []
            for item in lst:
                parts = item.split('.')
                order = int(parts[0])
                word_label = parts[1].strip().split(':')
                word = word_label[0]
                label = word_label[1] if len(word_label) > 1 else '0'
                parsed_data.append((order, word, label))
            
            # Adapted to DataFrame
            df = pd.DataFrame(parsed_data, columns=['order', 'word', 'label'])
            
            # Rename in the following format
            df.rename(columns={'word': f'word_{list_number}', 'label': f'label_{list_number}'}, inplace=True)
            return df

        # Load existing CSV file to append new columns
        csv_filename = f'../outputs/{output_folder}/csv/raw/(conventional_{output_folder.split("_")[1]}){output_type}.csv'
        existing_df = pd.read_csv(csv_filename)

        # Add as new columns into DataFrame
        for list_number, lst in merged_texts.items():
            df = process_list(lst, list_number)
            # Add as word_i column and label_i column
            df = df[['word_' + str(list_number), 'label_' + str(list_number)]]
            # Merge into the DataFrame
            existing_df = pd.concat([existing_df, df], axis=1)

        # Update the CSV file with the appended data
        existing_df.to_csv(csv_filename, index=False)
