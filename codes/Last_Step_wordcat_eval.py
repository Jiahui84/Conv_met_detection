# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import os

# This function processes a single CSV file, categorizing rows into 
# True Positives (TP), False Positives (FP), and False Negatives (FN) based on the 'DELMET' and label values.
def process_csv(file_path, output_base, subfolder):
    df = pd.read_csv(file_path)

    # Prepare empty DataFrames with columns for words, word categories, and context for up to 25 labels.
    column_labels = []
    for i in range(1, 26):
        column_labels.extend([f'word_{i}', f'wordcat_{i}', f'context_{i}'])

    tp_df = pd.DataFrame(columns=column_labels)  # True Positives
    fp_df = pd.DataFrame(columns=column_labels)  # False Positives
    fn_df = pd.DataFrame(columns=column_labels)  # False Negatives

    # Process each row in the DataFrame
    for _, row in df.iterrows():
        delmet = row['DELMET']  # Ground truth label
        wordcat = row['wordcat']  # Category of the word
        context = row['context']  # Context of the word

        tp_row = {}
        fp_row = {}
        fn_row = {}

        # Analyze predictions for each of the 25 labels
        for i in range(1, 26):
            word_i = row[f'word_{i}']  # Word corresponding to the current label
            label_i = row[f'label_{i}']  # Predicted label

            # Categorize the results into TP, FP, or FN based on ground truth and prediction
            if delmet == 1 and label_i == 1:
                tp_row[f'word_{i}'] = word_i
                tp_row[f'wordcat_{i}'] = wordcat
                tp_row[f'context_{i}'] = context
            elif delmet == 1 and label_i == 0:
                fn_row[f'word_{i}'] = word_i
                fn_row[f'wordcat_{i}'] = wordcat
                fn_row[f'context_{i}'] = context
            elif delmet == 0 and label_i == 1:
                fp_row[f'word_{i}'] = word_i
                fp_row[f'wordcat_{i}'] = wordcat
                fp_row[f'context_{i}'] = context

        # Append the categorized row to the corresponding DataFrame
        if tp_row:
            tp_df = pd.concat([tp_df, pd.DataFrame([tp_row])], ignore_index=True)
        if fp_row:
            fp_df = pd.concat([fp_df, pd.DataFrame([fp_row])], ignore_index=True)
        if fn_row:
            fn_df = pd.concat([fn_df, pd.DataFrame([fn_row])], ignore_index=True)


    # Ensure output directories exist for TP, FP, and FN categories
    for category in ['TP', 'FP', 'FN']:
        os.makedirs(os.path.join(output_base, category, subfolder), exist_ok=True)

    # Save the categorized results to corresponding CSV files in the output directory
    tp_df.to_csv(os.path.join(output_base, 'TP', subfolder, os.path.basename(file_path)), index=False)
    fp_df.to_csv(os.path.join(output_base, 'FP', subfolder, os.path.basename(file_path)), index=False)
    fn_df.to_csv(os.path.join(output_base, 'FN', subfolder, os.path.basename(file_path)), index=False)

# This function processes all relevant CSV files in a list of folders,
# applying the 'process_csv' function to each file.
def batch_process_folders(base_folder, output_folders, new_output_base):
    for folder in output_folders:
        # Construct the path to the folder containing the CSV files
        target_path = os.path.join(base_folder, f"output_{folder}", 'csv', 'sorted')
        print(f"Processing folder: {target_path}")
        for file_name in os.listdir(target_path):
            if file_name.endswith('.csv'):
                # Process each CSV file found in the folder
                process_csv(os.path.join(target_path, file_name), new_output_base, str(folder))

# Calling the batch processing function to handle multiple folders of CSV files.
base_folder = '../outputs/'
output_folders = [0, 1, 5, 10]  # These are the suffixes for the folders to be processed
new_output_base = '../analysis/wordcat error analysis/raw'  # Replace with your desired output path
batch_process_folders(base_folder, output_folders, new_output_base)

# This function processes statistical analysis of the word categories (wordcat) across multiple CSV files.
def process_stats(input_base, output_base):
    # Ensure the output directories exist for each category (TP, FP, FN) and each subfolder (0, 1, 5, 10).
    categories = ['TP', 'FP', 'FN']
    subfolders = ['0', '1', '5', '10']
    for category in categories:
        for subfolder in subfolders:
            os.makedirs(os.path.join(output_base, category, subfolder), exist_ok=True)

    # Process each CSV file within each subfolder of each category.
    for category in categories:
        for subfolder in subfolders:
            input_path = os.path.join(input_base, category, subfolder)
            output_path = os.path.join(output_base, category, subfolder)

            for file_name in os.listdir(input_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(input_path, file_name)
                    df = pd.read_csv(file_path)

                    # Initialize a DataFrame to store all statistical results for the current file.
                    stats_df = pd.DataFrame()

                    # Process each word category column (wordcat_i) from 1 to 25.
                    for i in range(1, 26):  # Assuming there are 25 wordcat_i columns
                        col_name = f'wordcat_{i}'
                        if col_name in df.columns:
                            # Split the values in the wordcat column by '+', count the occurrences, and store them.
                            split_words = df[col_name].dropna().apply(lambda x: x.split('+')).explode().str.strip()
                            word_count = split_words.value_counts().rename(f'frequency_{i}')

                            # Merge the word count with the stats DataFrame.
                            stats_df = pd.concat([stats_df, word_count], axis=1)

                    # Save the statistical results to a CSV file if the DataFrame is not empty.
                    if not stats_df.empty:
                        stats_df.fillna(0, inplace=True)
                        stat_file_name = f"{os.path.splitext(file_name)[0]}_stats.csv"
                        stats_df.to_csv(os.path.join(output_path, stat_file_name), index=True)

# Calling the function to process statistical analysis
input_base = '../analysis/wordcat error analysis/raw'  # Input directory containing the raw data
output_base = '../analysis/wordcat error analysis/stats'  # Output directory to save the statistics
process_stats(input_base, output_base)

# This function aggregates statistics by calculating the average frequency of words in 
# the word categories (wordcat) across multiple CSV files for different categories and subfolders.
def aggregate_stats(input_base, output_base):
    # Define the categories (TP, FP, FN) and the subfolders (0, 1, 5, 10) to process.
    categories = ['TP', 'FP', 'FN']
    subfolders = ['0', '1', '5', '10']

    # Iterate through each category and subfolder.
    for category in categories:
        for subfolder in subfolders:
            input_path = os.path.join(input_base, category, subfolder)
            output_path = os.path.join(output_base, category, subfolder)
            os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists.

            # Find all files in the input path that end with '_stats.csv'
            files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('_stats.csv')]
            if not files:
                print(f"No files to process in {input_path}")
                continue

            dataframes = []
            for file in files:
                # Load the CSV without a header and set the first column as the index.
                df = pd.read_csv(file, header=None, skiprows=[0], names=['wordcat'] + [f'frequency_{i}' for i in range(1, 26)])
                df.set_index('wordcat', inplace=True)
                dataframes.append(df)

            if not dataframes:
                print("No dataframes to process after loading files.")
                continue

            # Combine all dataframes horizontally (side by side).
            combined_df = pd.concat(dataframes, axis=1)

            # Prepare a DataFrame to store the average frequencies.
            mean_df = pd.DataFrame(index=combined_df.index)

            # Calculate the mean for each frequency column across all combined dataframes.
            for i in range(1, 26):  # Assuming there are 25 frequency columns
                specific_freq_cols = [col for col in combined_df.columns if col == f'frequency_{i}']
                if specific_freq_cols:
                    mean_df[f'average_frequency_{i}'] = combined_df[specific_freq_cols].mean(axis=1, skipna=True)

            # Save the calculated mean frequencies to a new CSV file.
            if not mean_df.empty:
                mean_file_name = f"{subfolder}_mean_stats.csv"
                mean_df.to_csv(os.path.join(output_path, mean_file_name), index=True)
                print(f"Saved mean statistics to {os.path.join(output_path, mean_file_name)}")
            else:
                print(f"Mean dataframe is empty, no data to save for {subfolder}.")

# Call the function to aggregate statistics
input_base = '../analysis/wordcat error analysis/stats'  # Directory containing the input statistics files
output_base = '../analysis/wordcat error analysis/average_stats'  # Directory to save the aggregated average statistics
aggregate_stats(input_base, output_base)

# This function plots the average values for each word category across different subfolders (shots) and categories (TP, FP, FN).
def plot_averages(base_dir, categories, subfolders):
    # Iterate through each category (TP, FP, FN)
    for category in categories:
        wordcat_data = {}  # Dictionary to store data for all wordcats in this category

        # Load data from each subfolder (shot setting)
        for subfolder in subfolders:
            folder_path = os.path.join(base_dir, category, subfolder)
            file_path = os.path.join(folder_path, f"{subfolder}_mean_stats.csv")
            
            # Check if the file exists and load it
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df.set_index('wordcat', inplace=True)
                
                # Store data in dictionary
                for wordcat in df.index:
                    if wordcat not in wordcat_data:
                        wordcat_data[wordcat] = {}
                    wordcat_data[wordcat][subfolder] = df.loc[wordcat]

        # Plot data for each wordcat
        for wordcat, data in wordcat_data.items():
            plt.figure(figsize=(12, 6))  # Create a new figure
            markers = ['o', 'v', '^', '<', '>']  # Different markers for different shots
            colors = ['blue', 'green', 'red', 'cyan']  # Colors for each shot
            shot_labels = ['0shot', '1shot', '5shot', '10shot']  # Labels for the legend

            # Plot each subfolder's data
            for i, subfolder in enumerate(subfolders):
                if subfolder in data:
                    plt.plot(range(1, 26), data[subfolder], marker=markers[i % len(markers)],
                             color=colors[i % len(colors)], label=f'{shot_labels[i]} ({category})')

                    # Add text labels for each point
                    for j, value in enumerate(data[subfolder]):
                        plt.text(j + 1, value, f'{value:.2f}', fontsize=8, ha='right')

            # Customize and save the plot
            plt.title(f'Average Values for {wordcat} - {category}')
            plt.xlabel('Prompt Number')
            plt.xticks(range(1, 26))  # Set x-ticks to show each frequency category number
            plt.ylabel('Average Value', fontsize=16)
            plt.legend()
            plt.grid(True)
            save_path = os.path.join(base_dir, category, f'{wordcat}_averages.png')
            plt.savefig(save_path)  # Save the plot for each wordcat
            plt.close()

# Define paths and categories
base_dir = '../analysis/wordcat error analysis/average_stats'  # Base directory containing the mean statistics
categories = ['TP', 'FP', 'FN']  # Categories to process
subfolders = ['0', '1', '5', '10']  # Subfolders (shot settings)

# Generate plots for average values
plot_averages(base_dir, categories, subfolders)