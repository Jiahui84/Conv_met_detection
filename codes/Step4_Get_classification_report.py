import argparse
import re
import os
from glob import glob
import pandas as pd
import csv
import json
from sklearn.metrics import classification_report

# Initialize cumulative metrics dictionaries
cumulative_metrics_class_0 = {
    'Accuracy': 0,
    'F1': 0,
    'Precision': 0,
    'Recall': 0,
    'Support': 0,
    'Files_Count': 0
}

cumulative_metrics_class_1 = {
    'Accuracy': 0,
    'F1': 0,
    'Precision': 0,
    'Recall': 0,
    'Support': 0,
    'Files_Count': 0
}

# Modify input file paths based on output and conventional numbers
def modify_input_file(input_file, output_number, conventional_number):
    new_input_file = re.sub(r'output_x', f'output_{output_number}', input_file)
    new_input_file = re.sub(r'conventional_x', f'conventional_{conventional_number}', new_input_file)
    return new_input_file

# Process files with a specific output number
def process_with_output_number(output_number, input_file, log_file):
    # Modify output directories based on the output_number
    output_directory = f'../outputs/output_{output_number}/results_log/'
    output_directory_class_0 = f'../outputs/secret/output_{output_number}/csv/results/class_0/'
    output_directory_class_1 = f'../outputs/secret/output_{output_number}/csv/results/class_1/'
    output_directory_summary_0 = f'../outputs/secret/output_{output_number}/csv/results/class_0/overall'
    output_directory_summary_1 = f'../outputs/secret/output_{output_number}/csv/results/class_1/overall'

    print(f"Processing for output number: {output_number}")
    print("Modified input file:", input_file)
    print("Log output directory:", log_file)

    # Call the main processing logic for this input_file
    main(input_file, log_file, output_number)


# name the output files based on the name pattern of the input files
def construct_csv_filenames(base_filepath, class_0_dir, class_1_dir, summary_0_dir, summary_1_dir):
    base_filename = os.path.basename(base_filepath)

    # extract the pattern part (within brackets) for summary filenames
    pattern_start = base_filename.find('(')
    pattern_end = base_filename.find(')')
    if pattern_start != -1 and pattern_end != -1:
        pattern_part = base_filename[pattern_start:pattern_end + 1]  # Include brackets
    else:
        print(f"Pattern not found in filename {base_filename}")
        return None, None, None, None

    # for class_0 and class_1 filenames, use the part before '_output_sorted.csv'
    detailed_part = base_filename.split('_output_sorted.csv')[0]

    # construct individual class CSV filenames
    class_0_csv_filename = os.path.join(class_0_dir, f'{detailed_part}_class_0.csv')
    class_1_csv_filename = os.path.join(class_1_dir, f'{detailed_part}_class_1.csv')

    # construct summary CSV filenames for average results
    summary_class_0_csv_filename = os.path.join(summary_0_dir, f'{pattern_part}_class_0_average.csv')
    summary_class_1_csv_filename = os.path.join(summary_1_dir, f'{pattern_part}_class_1_average.csv')

    return class_0_csv_filename, class_1_csv_filename, summary_class_0_csv_filename, summary_class_1_csv_filename

# accumulates classification metrics and support counts for multiple files, updating a dictionary
def calculate_cumulative_metrics(metrics, class_report, overall_accuracy, is_class_0):
    metrics['Accuracy'] += overall_accuracy
    metrics['F1'] += class_report['f1-score']
    metrics['Precision'] += class_report['precision']
    metrics['Recall'] += class_report['recall']
    metrics['Support'] += class_report['support']
    metrics['Files_Count'] += 1
    return metrics

# calculates the average of multiple sets of metrics stored in a dictionary, where each set corresponds to a label(prompt)
def calculate_average_metrics(metrics_dict):
    # initialize a new dictionary to store averages
    average_metrics = {}
    for label, metrics_list in metrics_dict.items():
        # make sure metrics_list is not empty
        if not metrics_list:
            continue
        label_averages = {key: 0 for key in metrics_list[0].keys() if key != 'Files_Count'}
        files_count = len(metrics_list)  # use length of list as file count
        for metrics in metrics_list:
            for key in label_averages:
                label_averages[key] += metrics[key]
        # calculate the average
        for key in label_averages:
            label_averages[key] /= files_count
        average_metrics[label] = label_averages
    return average_metrics

# specify the wirte in format of the average summary csv files
def write_summary_csv(averaged_metrics_dict, summary_csv_filename):
    with open(summary_csv_filename, 'w', newline='') as file:
        # If the dictionary is not empty, get the fieldnames from the first entry
        if averaged_metrics_dict:
            first_key = next(iter(averaged_metrics_dict))
            fieldnames = ['Label_Column'] + list(averaged_metrics_dict[first_key].keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for label, metrics in averaged_metrics_dict.items():
                # Insert label at the beginning of the metrics
                row = {'Label_Column': label}
                row.update(metrics)
                writer.writerow(row)

# Main function for processing
def main(fn_pattern, output_filename, output_number):
    global cumulative_metrics_class_0
    global cumulative_metrics_class_1

    label_data_dict_class_0 = {}
    label_data_dict_class_1 = {}

    if output_filename is None:
        output_filename = 'default_log.log'

    output_file = os.path.join(f'../outputs/output_{output_number}/results_log/', output_filename)
    print(output_file)

    # Use output_number instead of number
    modified_fn_pattern = modify_input_file(fn_pattern, output_number, output_number)
    print(f"Processing for output number: {output_number}")
    print(f"Modified input file: {modified_fn_pattern}")

    matched_files = glob(modified_fn_pattern)
    if not matched_files:
        print(f"No files found for pattern: {modified_fn_pattern}")
    else:
        print(f"Files matching pattern {modified_fn_pattern}: {matched_files}")
    
    for file_path in matched_files:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")

        # Filter out deliberate metaphors labelled with 2
        df = df[df['DELMET'] != 2]

        # Automatically detect all columns starting with "label_"
        pred_cols = [col for col in df.columns if col.startswith('label_')]

        # Construct CSV filenames for class 0 and class 1
        class_0_csv_filename, class_1_csv_filename, summary_class_0_csv_filename, summary_class_1_csv_filename = construct_csv_filenames(
            file_path, 
            f'../outputs/output_{output_number}/csv/results/class_0/', 
            f'../outputs/output_{output_number}/csv/results/class_1/', 
            f'../outputs/output_{output_number}/csv/results/class_0/overall', 
            f'../outputs/output_{output_number}/csv/results/class_1/overall'
        )


        if not class_0_csv_filename:
            continue  # Skip if filename pattern not matched

        with open(class_0_csv_filename, 'w', newline='') as class_0_csvfile, \
             open(class_1_csv_filename, 'w', newline='') as class_1_csvfile:
            print(f"Writing class 0 results to {class_0_csv_filename}")
            print(f"Writing class 1 results to {class_1_csv_filename}")


            writer_0 = csv.DictWriter(class_0_csvfile, fieldnames=['Label_Column', 'Accuracy', 'F1', 'Precision', 'Recall', 'Support'])
            writer_1 = csv.DictWriter(class_1_csvfile, fieldnames=['Label_Column', 'Accuracy', 'F1', 'Precision', 'Recall', 'Support'])
            writer_0.writeheader()
            writer_1.writeheader()

            for pred_col in pred_cols:
                gt_l = list(df['DELMET'])
                pred_l = list(df[pred_col])

                gt_l = list(map(str, gt_l))
                pred_l = list(map(str, pred_l))

                report = classification_report(gt_l, pred_l, output_dict=True)

                row_0 = report['0']
                row_1 = report['1']
                overall_accuracy = report['accuracy']

                cumulative_metrics_class_0 = calculate_cumulative_metrics(cumulative_metrics_class_0, report['0'], overall_accuracy, True)
                cumulative_metrics_class_1 = calculate_cumulative_metrics(cumulative_metrics_class_1, report['1'], overall_accuracy, False)

                csv_row_0 = {
                    'Label_Column': pred_col,
                    'Accuracy': report['accuracy'],
                    'F1': row_0['f1-score'],
                    'Precision': row_0['precision'],
                    'Recall': row_0['recall'],
                    'Support': row_0['support']
                }

                csv_row_1 = {
                    'Label_Column': pred_col,
                    'Accuracy': report['accuracy'],
                    'F1': row_1['f1-score'],
                    'Precision': row_1['precision'],
                    'Recall': row_1['recall'],
                    'Support': row_1['support']
                }

                writer_0.writerow(csv_row_0)
                writer_1.writerow(csv_row_1)

                with open(output_file, 'a') as f:
                    f.write(f"File: {file_path}, Label Column: {pred_col}\n")
                    f.write(json.dumps(report, indent=4) + "\n\n")

                # save to dictionary
                if pred_col not in label_data_dict_class_0:
                    label_data_dict_class_0[pred_col] = []
                label_data_dict_class_0[pred_col].append(row_0)  # use row_0

                if pred_col not in label_data_dict_class_1:
                    label_data_dict_class_1[pred_col] = []
                label_data_dict_class_1[pred_col].append(row_1)  # use row_1

            # Calculate and write averages
            label_averages_class_0 = calculate_average_metrics(label_data_dict_class_0)
            label_averages_class_1 = calculate_average_metrics(label_data_dict_class_1)

            write_summary_csv(label_averages_class_0, summary_class_0_csv_filename)
            write_summary_csv(label_averages_class_1, summary_class_1_csv_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process file paths dynamically.")
    parser.add_argument("--input_file", type=str, help="Path to the sorted CSV file")
    parser.add_argument("--log_file", type=str, default="conventional_results.log", help="Name of the log file")
    args = parser.parse_args()

    if args.input_file:
        match = re.search(r'output_(\d+).*conventional_(\d+)', args.input_file)
        if match:
            output_number = match.group(1)
            conventional_number = match.group(2)
            print(f"Using command-line provided paths: output_{output_number} and conventional_{conventional_number}")
            process_with_output_number(output_number, args.input_file, args.log_file)
        else:
            print("Error: The provided input_file does not contain both 'output_' and 'conventional_' patterns.")
    else:
        print("No input_file provided, using default loop over [0, 1, 5, 10]")
        template_input_file = "../outputs/output_x/csv/sorted/(conventional_x)*_output_sorted.csv"
        numbers = [0, 1, 5, 10]

        for output_number in numbers:  # Change variable name here
            modified_input_file = modify_input_file(template_input_file, output_number, output_number)
            process_with_output_number(output_number, modified_input_file, args.log_file)
