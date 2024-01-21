from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from glob import glob
import sys
import os

# specify the output path for logs
output_directory = '../outputs/output_0/results_log/'

def main(fn_pattern, output_filename, class_arg):
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
        # df['DELMET'] = df['DELMET'].replace(2, 1)

        # for the prompts only aim at detecting noun,verb,adjective (conNVAJ, lexiNVAJ and metNVAJ), use the two lines below to filter rows in wordcat column that do not belong to noun, verb, adjective
        # excluded_tags = ['CJ', 'EX', 'AV', 'AT', 'PN', 'PR', 'DP', 'DT', 'TO', 'XX', 'CR', 'OR', 'UN', 'ZZ']
        # wordcat_filtered = df[~df['wordcat'].isin(excluded_tags)]

        # Automatically detect all the columns starting with "label_"
        pred_cols = [col for col in df.columns if col.startswith('label_')]
        
        for pred_col in pred_cols:
            gt_l = list(df['DELMET'])  # true label column
            pred_l = list(df[pred_col])  # predict label column

            if class_arg is not None:
                # Filter data based on the specified class
                filtered_gt_l = [x for i, x in enumerate(gt_l) if x == class_arg or pred_l[i] == class_arg]
                filtered_pred_l = [x for i, x in enumerate(pred_l) if gt_l[i] == class_arg or x == class_arg]

                report = classification_report(filtered_gt_l, filtered_pred_l, labels=[class_arg])
                tn, fp, fn, tp = confusion_matrix(filtered_gt_l, filtered_pred_l).ravel()
            else:
                report = classification_report(gt_l, pred_l)
                tn, fp, fn, tp = confusion_matrix(gt_l, pred_l).ravel()

                # calculate the number of NVAJ in true labels and in pre labels
                # NVAJ_total = len(wordcat_filtered)
                # NVAJ_detected = wordcat_filtered[wordcat_filtered[pred_col] == 1].shape[0]


            # print confusion_matrix and classification_report 
            print(f"File: {file_path}, Label Column: {pred_col}")
            print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")
            # when it's resluts for the prompts only aiming at detecting noun,verb,adjective, can use the line below to see the number of NVAJ in true labels and in pre labels
            # print(f"NVAJ Total: {NVAJ_total}, NVAJ Detected: {NVAJ_detected}")
            print(report)
            print("\n")

            # write the results inro log
            with open(output_file, 'a') as f:
                f.write(f"File: {file_path}, Label Column: {pred_col}\n")
                f.write(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}\n")
                # when it's resluts for the prompts only aiming at detecting noun,verb,adjective, can use the line below to see the number of NVAJ in true labels and in pre labels
                # f.write(f"NVAJ Total: {NVAJ_total}, NVAJ Detected: {NVAJ_detected}")
                f.write(report + "\n\n")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python script.py file_pattern output_filename [class]")
        sys.exit(1)

    file_pattern = sys.argv[1]# a regex referring to all prediction files for one metaphor type, like "conventional_*.csv"
    output_filename = sys.argv[2]# get the output logs
    class_arg = None if len(sys.argv) == 3 else int(sys.argv[3])# get report for specific class (1 - conventional metaphors, 0 - others)
    if class_arg is None:
        output_directory = '../outputs/output_0/results_log/both_classes/'
    elif class_arg == 1:
        output_directory = '../outputs/output_0/results_log/class_1/'
    elif class_arg == 0:
        output_directory = '../outputs/output_0/results_log/class_0/'
    else:
        print("Invalid class argument. Please specify 0, 1, or leave blank for both.")
        sys.exit(1)

    main(file_pattern, output_filename, class_arg)