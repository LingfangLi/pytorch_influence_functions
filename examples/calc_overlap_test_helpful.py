import logging
import os

import pandas as pd
import pytorch_influence_functions as ptif
def calc_overlap_test_helpful(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Preprocess the 'helpful_samples' column to split into sets
    data['set_helpful_samples'] = data['helpful_samples'].apply(lambda x: set(x.split('|')))

    data['set_helpful_samples'].head()

    # Since 'test_sample' is considered a set with a single element, we create a corresponding set for it
    data['set_test_sample'] = data['test_sample'].apply(lambda x: set([x.strip()]))

    # Calculate overlap for each row
    data['overlap'] = data.apply(lambda row: len(row['set_test_sample'].intersection(row['set_helpful_samples'])) / \
                                 float(len(row['set_test_sample'].union(row['set_helpful_samples']))), axis=1)

    # Calculate average overlap
    average_overlap = data['overlap'].mean()

    return average_overlap


# Define the function to calculate the overlap for a single row
def calc_row_overlap(test_sample, helpful_sample):
    # Split the test sample and helpful samples into sets of unique words
    test_sample_set = set(test_sample.lower().split())
    helpful_sample_set = set(helpful_sample.lower().replace('|', ' ').split())

    # Calculate the overlap and return it
    overlap = len(test_sample_set.intersection(helpful_sample_set))
    return overlap

# Define the function to calculate the average overlap for a single file
def calc_file_overlap(csv_path):
    df = pd.read_csv(csv_path)
    total_overlap = 0
    num_rows = 0

    # Iterate over each row and calculate the overlap
    for _, row in df.iterrows():
        if pd.notnull(row['test_sample']) and pd.notnull(row['helpful_samples']):
            overlap = calc_row_overlap(row['test_sample'], row['helpful_samples'])
            total_overlap += overlap
            num_rows += 1

    # Return the total overlap and number of rows for this file
    return total_overlap, num_rows

# Define the function to calculate the average overlap across all CSV files in the directory
# Update the function to walk through a directory and calculate the average overlap for all CSV files
def calc_average_overlap_in_directory(directory_path):
    total_overlap = 0
    total_rows = 0

    # Walk through the directory and process each CSV file
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                file_overlap, file_rows = calc_file_overlap(file_path)
                total_overlap += file_overlap
                total_rows += file_rows

    # Calculate the average overlap
    if total_rows > 0:
        average_overlap = total_overlap / total_rows
    else:
        average_overlap = None

    return average_overlap




if __name__ == "__main__":
    average_overlap=calc_average_overlap_in_directory('D:\OneDrive - The University of Liverpool\LLMs\InfluenceFunctions\output_sentence')

    # set the directory for log file
    ptif.init_logging('D:\OneDrive - The University of Liverpool\LLMs\InfluenceFunctions\logfile.log')
    logging.info(f'Average overlap: {average_overlap}')