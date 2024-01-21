from sklearn.metrics import classification_report
import pandas as pd
from glob import glob
import sys
import os
import csv
import json

# specify the output path for logs
output_directory = '../outputs/output_0/results_log/'

# specify the output path for CSV files for class 0 and class 1
output_directory_class_0 = '../outputs/output_0/csv/results/class_0/'
output_directory_class_1 = '../outputs/output_0/csv/results/class_1/'

# specify the output path for the average summary CSV files for class 0 and class 1
output_directory_summary_0 = '../outputs/output_0/csv/results/class_0/overall'
output_directory_summary_1 = '../outputs/output_0/csv/results/class_1/overall'

# initialize cumulative indicator dictionary
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



def main(fn_pattern, output_filename):
    
    # make sure the initialized cumulative indicator dictionaries are involved
    global cumulative_metrics_class_0
    global cumulative_metrics_class_1

    # initialize dictionaries to store data for each label
    label_data_dict_class_0 = {}
    label_data_dict_class_1 = {}

    if output_filename is None:
        # if filename is not provided, use the default name
        output_filename = 'default_log.log'
    
    output_file = os.path.join(output_directory, output_filename)
    print(output_file)
    
    for file_path in glob(fn_pattern):
        df = pd.read_csv(file_path, encoding="ISO-8859-1")

        # filter out all the delberate metaphors labelled with 2
        df = df[df['DELMET'] != 2]
        # for the prompts only aim at detecting metaphors (metaohor and metNVAJ), Use the code below rather than the last code
        #df['DELMET'] = df['DELMET'].replace(2, 1)

        # automatically detect all the columns starting with "label_"
        pred_cols = [col for col in df.columns if col.startswith('label_')]

        # use the new function to construct CSV filenames
        class_0_csv_filename, class_1_csv_filename, summary_class_0_csv_filename, summary_class_1_csv_filename = construct_csv_filenames(
            file_path, output_directory_class_0, output_directory_class_1, output_directory_summary_0, output_directory_summary_1)

            
        if not class_0_csv_filename:
            continue  # skip this file if the filename pattern is not matched

        # define the fieldnames for the CSV files
        fieldnames = [
            'Label_Column', 'Accuracy', 
            'F1', 'Precision', 'Recall', 'Support'
        ]

        # open CSV files for writing
        with open(class_0_csv_filename, 'w', newline='') as class_0_csvfile, \
            open(class_1_csv_filename, 'w', newline='') as class_1_csvfile:

            writer_0 = csv.DictWriter(class_0_csvfile, fieldnames=fieldnames)
            writer_1 = csv.DictWriter(class_1_csvfile, fieldnames=fieldnames)
            writer_0.writeheader()
            writer_1.writeheader()
        
            for pred_col in pred_cols:
                gt_l = list(df['DELMET'])  # true label column
                pred_l = list(df[pred_col])  # predict label column

                report = classification_report(gt_l, pred_l, output_dict=True)
                # convert report dictionary to string
                report_str = json.dumps(report, indent=4)
                report_original = classification_report(gt_l, pred_l)

                # print and classification_report 
                print(f"File: {file_path}, Label Column: {pred_col}")
                print(report_original)
                print("\n")

                # check and adjust if there are more classes
                row_0 = report['0']
                row_1 = report['1']
                overall_accuracy = report['accuracy']  # accuracy is overall, not per class

                # check that report has the expected keys
                if '0' not in report or '1' not in report:
                    print(f"Expected class keys not found in report for {pred_col}.")
                    continue

                # update cumulative metrics using reporting data
                cumulative_metrics_class_0 = calculate_cumulative_metrics(
                    cumulative_metrics_class_0, report['0'], overall_accuracy, True
                    )
                cumulative_metrics_class_1 = calculate_cumulative_metrics(
                    cumulative_metrics_class_1, report['1'], overall_accuracy, False
                    )


                # build a dictionary for writing class 0 CSV
                csv_row_0 = {
                    'Label_Column': pred_col,
                    'Accuracy': report['accuracy'],  # get from classification report
                    'F1': row_0['f1-score'],
                    'Precision': row_0['precision'],
                    'Recall': row_0['recall'],
                    'Support': row_0['support']
                    }

                # build a dictionary for writing class 1 CSV
                csv_row_1 = {
                    'Label_Column': pred_col,
                    'Accuracy': report['accuracy'],
                    'F1': row_1['f1-score'],
                    'Precision': row_1['precision'],
                    'Recall': row_1['recall'],
                    'Support': row_1['support']
                    }

                # write into CSV
                writer_0.writerow(csv_row_0)
                writer_1.writerow(csv_row_1)

                # write the results inro log
                with open(output_file, 'a') as f:
                    f.write(f"File: {file_path}, Label Column: {pred_col}\n")
                    f.write(report_str + "\n\n")

                # add scores to class_0 and class_1
                label_data_0 = {
                    'Accuracy': report['accuracy'],
                    'F1': report['0']['f1-score'],
                    'Precision': report['0']['precision'],
                    'Recall': report['0']['recall'],
                    'Support': report['0']['support']
                }

                label_data_1 = {
                    'Accuracy': report['accuracy'],
                    'F1': report['1']['f1-score'],
                    'Precision': report['1']['precision'],
                    'Recall': report['1']['recall'],
                    'Support': report['1']['support']
                }

                if pred_col not in label_data_dict_class_0:
                    label_data_dict_class_0[pred_col] = []
                label_data_dict_class_0[pred_col].append(label_data_0)

                if pred_col not in label_data_dict_class_1:
                    label_data_dict_class_1[pred_col] = []
                label_data_dict_class_1[pred_col].append(label_data_1)
    
    # after processing all the input files, calculate their average scores
    label_averages_class_0 = calculate_average_metrics(label_data_dict_class_0)
    label_averages_class_1 = calculate_average_metrics(label_data_dict_class_1)

    # calculate the average for every label and write into csv
    write_summary_csv(label_averages_class_0, summary_class_0_csv_filename)
    write_summary_csv(label_averages_class_1, summary_class_1_csv_filename)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python Get_classification_report.py file_pattern output_filename")
        sys.exit(1)

    file_pattern = sys.argv[1]  # # a regex referring to all prediction files for one metaphor type, like "conventional_*.csv"
    output_filename = sys.argv[2]  # get the output logs
    main(file_pattern, output_filename)