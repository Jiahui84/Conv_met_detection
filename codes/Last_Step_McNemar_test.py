#The experiments with 0-shot, 1-shot, 5-shot, and 10-shot were each run three times, so we also performed three comparisons when comparing different shots.
# Load libraries
import pandas as pd
import os
import re
from statsmodels.stats.contingency_tables import mcnemar
from itertools import combinations
from pathlib import Path

## Comparison of same prompt in different shot prompting
# Define base folder path
base_folder = '../outputs/output_{}/csv/sorted/'

# Define folder paths
folders = [base_folder.format(i) for i in [0, 1, 5, 10]]

# Get the list of files in each folder
def get_files(folder):
    return sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

# Get the common file names
common_files = ['first_output_sorted.csv', 'second_output_sorted.csv', 'third_output_sorted.csv']

# Define significance level
alpha = 0.05

def compare_files(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Ensure ground truth columns are the same
    assert all(df1['DELMET'] == df2['DELMET']), "Ground truth labels do not match."

    results = []

    # Perform McNemar's test for each label_i
    for i in range(1, 26):
        label_col = f'label_{i}'

        # Create 2x2 confusion matrix
        b = sum((df1[label_col] == df1['DELMET']) & (df2[label_col] != df2['DELMET']))
        c = sum((df1[label_col] != df1['DELMET']) & (df2[label_col] == df2['DELMET']))

        # Conduct McNemar's test
        table = [[0, b], [c, 0]]  # Only b and c are needed
        result = mcnemar(table, exact=True)

        # Determine if the difference is significant
        significance = 'significant' if result.pvalue < alpha else 'not significant'

        # Save results
        results.append((label_col, result.pvalue, significance))

    return results

# Create log file
log_file = '../analysis/McNemar test/Comparison_across_shots/mcnemar_results_same_prompt_across_shots.log'

# Iterate through each pair of folders and compare files
with open(log_file, 'w') as f:
    for folder1, folder2 in combinations(folders, 2):
        files1 = get_files(folder1)
        files2 = get_files(folder2)
        
        f.write(f'Comparing {folder1} and {folder2}:\n')

        for common_file in common_files:
            file1_list = [f for f in files1 if common_file in f]
            file2_list = [f for f in files2 if common_file in f]
            
            if file1_list and file2_list:
                file1 = file1_list[0]
                file2 = file2_list[0]

                results = compare_files(os.path.join(folder1, file1), os.path.join(folder2, file2))

                # Write results to the log file
                f.write(f'Results for {common_file}:\n')
                for label, pvalue, significance in results:
                    f.write(f'{label}: p-value = {pvalue}, {significance}\n')
                f.write('\n')
            else:
                f.write(f'Files not found for {common_file}\n\n')

        f.write('---\n')  # Separate results for different folder combinations


def extract_significant_lines_with_structure(log_file): 
    significant_lines = []
    current_comparison = ''
    current_file = ''
    section_lines = []
    
    with open(log_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Detect comparison headers
            if line.startswith('Comparing'):
                # If there is a previous section, append it before starting a new one
                if section_lines:
                    significant_lines.extend(section_lines)
                    section_lines = []
                
                current_comparison = line.strip()  # Update the current comparison line
                section_lines.append(current_comparison)  # Start a new section with comparison line
            
            # Detect file headers
            elif line.startswith('Results for'):
                current_file = line.strip()  # Update the current file line
                section_lines.append(current_file)  # Add the file header to the section
            
            # Detect significant results
            elif 'significant' in line and 'not significant' not in line:
                section_lines.append(line.strip())  # Add only significant lines
            
            # Handle separator "---" between comparisons
            elif line.strip() == '---':
                if section_lines:
                    significant_lines.extend(section_lines)  # Add the last section
                    section_lines = []
                significant_lines.append('---')  # Add the separator
        
        # Append any remaining section after the loop
        if section_lines:
            significant_lines.extend(section_lines)
    
    return significant_lines


# Define your log file path
log_file = '../analysis/McNemar test/Comparison_across_shots/mcnemar_results_same_prompt_across_shots.log'

# Extract significant lines along with headers
significant_results = extract_significant_lines_with_structure(log_file)

# Print or save the results
for result in significant_results:
    print(result)

# Optionally, write the results to a new file
output_file = '../analysis/McNemar test/Comparison_across_shots/filtered_significant_results.log'
with open(output_file, 'w') as f_out:
    for result in significant_results:
        f_out.write(result + '\n')


## Comparison of different prompts in same shot prompting
# This function compares the performance of two classifiers (label1 and label2)  
# using McNemar's test to determine if there is a significant difference between them.
def make_contingency_table(data, label1, label2, ground_truth, alpha=.05):
    # Extract predictions of the two classifiers and the ground truth
    results_1 = data[label1]
    results_2 = data[label2]
    ground_truth = data[ground_truth]
    
    # Ensure that all inputs have the same length
    assert len(results_1) == len(results_2) == len(ground_truth), "Error: incompatible sizes."
    
    # Initialize a contingency table for counting occurrences of different outcomes
    Results = {(1, 1): 0, (0, 1): 0, (1, 0): 0, (0, 0): 0}
    performance_1 = sum(results_1 == ground_truth)  # Correct predictions by classifier 1
    performance_2 = sum(results_2 == ground_truth)  # Correct predictions by classifier 2
    
    # Populate the contingency table based on comparison of classifier predictions and ground truth
    for i in range(len(results_1)):
        score_1 = int(results_1[i] == ground_truth[i])
        score_2 = int(results_2[i] == ground_truth[i])
        Results[(score_1, score_2)] += 1
    
    # Create a 2x2 matrix from the contingency results
    table = [[Results[(1, 1)], Results[(1, 0)]],
             [Results[(0, 1)], Results[(0, 0)]]]
    
    # Perform McNemar's test to evaluate the significance of the difference between the two classifiers
    m = mcnemar(table, exact=False)
    
    # Determine if the difference is statistically significant based on the p-value
    significant = "Yes" if m.pvalue <= alpha else "No"
    return (f"{label1} vs {label2}", m.statistic, m.pvalue, significant)

# This function processes each folder containing the results of different shot settings.
# It reads CSV files, compares classifier performance using McNemar's test, 
# and saves the results into a new CSV file.

def process_folder():
    # Define the base folder path for input files
    base_folder = '../outputs/'
    
    # Iterate through folders corresponding to different shot settings
    for subdir in ['output_0', 'output_1', 'output_5', 'output_10']:
        full_path = Path(base_folder) / subdir / 'csv' / 'sorted'
        save_path = Path('../analysis/McNemar test/Comparison_between_prompts_in_different_shots') / subdir
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Process each CSV file in the folder
        for csv_file in full_path.glob('*.csv'):
            data = pd.read_csv(csv_file)
            results = []
            
            # Assuming 'DELMET' is the ground truth column and label columns are label_1 to label_25
            classifier_labels = [f'label_{i}' for i in range(1, 26)]
            
            # Compare each pair of classifiers using McNemar's test
            for (label1, label2) in combinations(classifier_labels, 2):
                result = make_contingency_table(data, label1, label2, 'DELMET')
                results.append(result)
            
            # Save the comparison results to a new CSV file
            results_df = pd.DataFrame(results, columns=['Label Pair', 'McNemar Statistic', 'P-Value', 'Significant'])
            result_file_name = csv_file.stem + '_results.csv'
            results_df.to_csv(save_path / result_file_name, index=False)

# Call the function to process the folders
process_folder()
