# Import needed libraries
import pandas as pd
import openai

# Prompt the user for their OpenAI API key
api_key = input("Please enter your OpenAI API key: ")
openai.api_key = api_key

# Let the user select which experiment run they are on
run_choice = input("Please select the experiment run (first, second, third): ").strip().lower()

# Map the user's choice to the correct folder name
if run_choice == "first":
    output_folder = "first_time_output"
elif run_choice == "second":
    output_folder = "second_time_output"
elif run_choice == "third":
    output_folder = "third_time_output"
else:
    print("Invalid choice. Please select 'first', 'second', or 'third'.")
    exit(1)

# Let the user select the shot number (0, 1, 5, 10) or leave blank for running all
shot_input = input("Please select the shot number (0, 1, 5, 10) or press Enter to run all: ").strip()

# Define the shot numbers to run
if shot_input:
    if shot_input not in ['0', '1', '5', '10']:
        print("Invalid choice. Please select 0, 1, 5, or 10.")
        exit(1)
    shot_numbers = [int(shot_input)]  # Run only the chosen shot
else:
    shot_numbers = [0, 1, 5, 10]  # Run all if no input

# Function for 0-shot logic
def run_0_shot():
    sentence_df = pd.read_csv('../data/corpus/sen4prompting.csv', usecols=['query_sentence'])
    prompts_df = pd.read_csv('../data/prompt_set/25_prompts_0shot.csv', usecols=['0 shot - conventional metaphor '])

    for prompt_index, prompt_row in prompts_df.iterrows():
        for sentence_index, sentence_row in sentence_df.iterrows():
            sentence = sentence_row['query_sentence']
            modified_prompt = prompt_row['0 shot - conventional metaphor '].replace("{text}", sentence)

            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[{"role": "user", "content": modified_prompt}],
                temperature=0.7,
                max_tokens=4000
            )
            generated_text = response['choices'][0]['message']['content'] if response['choices'] else 'No response'
            filename = f"../outputs/output_0/raw/{output_folder}/{prompt_index + 1}_{sentence_index + 1}.txt"
            
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(generated_text)

            print(f"Written to {filename}")

# Function for n-shot logic
def run_n_shot(shot_number):
    df = pd.read_csv('../data/corpus/example4prompting.csv')
    sentences, labels = select_and_combine(df, 'train', shot_number)
    sentence_df = pd.read_csv('../data/corpus/sen4prompting.csv', usecols=['query_sentence'], encoding='latin1')
    prompts_df = pd.read_csv('../data/prompt_set/25_prompts_nshot.csv', usecols=['1-n shot'])

    for prompt_index, prompt_row in prompts_df.iterrows():
        primary_output = ""
        for sentence, label in zip(sentences, labels):
            primary_output += prompt_row['1-n shot'].replace("{text}", sentence).replace("{label}", label) + "\n\n"

        for sentence_index, sentence_row in sentence_df.iterrows():
            combined_output = primary_output + prompt_row['1-n shot'].replace("{text}", sentence_row['query_sentence']).replace("{label}", "") + "\n\n"

            # Print the combined input for verification (if needed) 
            #print(f"Prompt Index: {prompt_index}, Sentence Index: {sentence_index}") 
            #print(combined_output) 

            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[{"role": "user", "content": combined_output}],
                temperature=0.7,
                max_tokens=4000
            )
            generated_text = response['choices'][0]['message']['content'] if response['choices'] else 'No response'
            filename = f"../outputs/output_{shot_number}/raw/{output_folder}/{prompt_index + 1}_{sentence_index + 1}.txt"

            with open(filename, 'w', encoding='utf-8') as file:
                file.write(generated_text)

            print(f"Written to {filename}")

# Helper function for selecting and combining n-shot examples
def select_and_combine(df, split_label, group_label):
    selected_df = df[(df['split'] == split_label) & (df['group'] == group_label)]
    sentences, labels = [], []
    for _, row in selected_df.iterrows():
        sentences.append(row['context'])
        labels.append(row['word_list'])
    return sentences, labels

# Run logic based on the user's input or automatically for all
for shot_number in shot_numbers:
    if shot_number == 0:
        print("Running 0-shot...")
        run_0_shot()
    else:
        print(f"Running {shot_number}-shot...")
        run_n_shot(shot_number)

print("All results have been written to text files.")
