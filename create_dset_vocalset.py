import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import subprocess
from tqdm import tqdm
import random


def prepare_data(root_dir):
    wav_files = []
    singers = []
    total_length = {}

    for root, dirs, files in os.walk(root_dir):
        for file in tqdm(files, desc="Splitting files"):
            if file.endswith(".wav"):
                tqdm.write(f"Splitting file: {os.path.join(root, file)}")
                singer = root.split('/')[4]
                
                # Split the WAV file into segments
                wav_length = split_wav_segments(os.path.join(root, file), output_dir="DataVocalSet", singer=singer)

                # Update the total length for the singer
                if singer in total_length:
                    total_length[singer] += wav_length
                else:
                    total_length[singer] = wav_length
    
    # Print the statistics
    print(f"Number of WAV files found: {len(wav_files)}")
    print(f"Number of unique singers found: {len(set(singers))}")
    print("Total length of data for each singer:")
    overall_length = 0
    for singer, length in total_length.items():
        print(f"{singer}: {round(length)} seconds")
        overall_length += length
    print(f"Overall length of data: {round(overall_length)} seconds")

    # Write the statistics to a file
    with open("DataVocalSet/statistics.txt", "w") as file:
        file.write(f"Number of WAV files found: {len(wav_files)}\n")
        file.write(f"Number of unique singers found: {len(set(singers))}\n")
        file.write(f"Overall length of data: {round(overall_length)} seconds\n")
        file.write("Total length of data for each singer:\n")
        for singer, length in total_length.items():
            file.write(f"{singer}: {round(length)} seconds\n")


def process_vocalset(prepared_data_dir):
    wav_files = []
    singers = []
    total_length = {}
    
    for root, dirs, files in os.walk(prepared_data_dir):
        for file in tqdm(files, desc="Processing files"):
            if file.endswith(".wav"):
                tqdm.write(f"Processing file: {os.path.join(root, file)}")
                wav_files.append(os.path.join(root, file))
                singer = root.split('/')[-1]
                singers.append(singer)
    
    # Perform label encoding
    label_encoder = LabelEncoder()
    singers_encoded = label_encoder.fit_transform(singers)
    
    # Get the mapping of string to integer
    mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    # Save the mapping to a text file
    np.savetxt("DataVocalSet/singer_to_label_mapping_vocalset.txt", np.array(list(mapping.items())), fmt="%s")
    
    # Zip the WAV paths and encoded labels as strings separated by "|"
    wav_labels = [f"{wav}|{label}" for wav, label in zip(wav_files, singers_encoded)]
    
    # Save the zipped data to a text file
    with open("DataVocalSet/vocalset_paths_labels.txt", "w") as file:
        file.write("\n".join(wav_labels))
    
    

def calculate_wav_length(wav_file):
    # Check if the file exists
    if not os.path.exists(wav_file):
        print(f"File not found: {wav_file}. Skipping...")
        return None
    
    # Run ffprobe command to get the duration of the WAV file
    command = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 '{wav_file}'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Parse the output to get the duration in seconds
    duration = float(result.stdout.strip())
    
    return duration


def split_wav_segments(wav_file, segment_length=5, output_dir="DataVocalSet", singer=""):
    # Check if the file exists
    if not os.path.exists(wav_file):
        print(f"File not found: {wav_file}. Skipping...")
        return None
    
    # Get the duration of the WAV file
    duration = calculate_wav_length(wav_file)
    
    # Calculate the number of segments
    num_segments = int(duration / segment_length)
    
    # Split the WAV file into segments using ffmpeg
    for i in range(num_segments):
        output_dir_singer = os.path.join(output_dir, singer)
        os.makedirs(output_dir_singer, exist_ok=True)

        # Get the number of files already in the directory
        num_files = len(os.listdir(output_dir_singer))
        # Increment the file number counter
        file_number = 1 + num_files
        
        output_file = os.path.join(output_dir_singer, f"{singer}_{file_number}.wav")
        
        command = f"ffmpeg -i {wav_file} -ss {i * segment_length} -t {segment_length} -c copy {output_file}"
        subprocess.run(command, shell=True)
    
    print(f"WAV file {wav_file} split into {num_segments} segments.")

    return duration


def create_splits_by_label(directory_path, eval_labels):
    train_set = []
    eval_set = []

    # Read the file vocalset_paths_labels.txt
    file_path = os.path.join(directory_path, "vocalset_paths_labels.txt")
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Split the lines into paths and labels
    data = [line.strip().split("|") for line in lines]

    # Get all unique labels
    all_labels = list(set([label for _, label in data]))

    # Determine the labels for the eval set
    if isinstance(eval_labels, int):
        eval_labels = random.sample(all_labels, eval_labels)
    elif isinstance(eval_labels, list):
        eval_labels = [label for label in eval_labels if label in all_labels]
    else:
        raise ValueError("Invalid eval_labels argument")

    # Split the data into train and eval sets
    for path, label in data:
        if label in eval_labels:
            eval_set.append((path, label))
        else:
            train_set.append((path, label))

    # Write train_set to train_list.txt
    train_file_path = os.path.join(directory_path, "train_list.txt")
    with open(train_file_path, "w") as train_file:
        for path, label in train_set:
            train_file.write(f"{path}|{label}\n")

    # Write eval_set to val_list.txt
    eval_file_path = os.path.join(directory_path, "val_list.txt")
    with open(eval_file_path, "w") as eval_file:
        for path, label in eval_set:
            eval_file.write(f"{path}|{label}\n")



def create_splits(directory_path, eval_fraction):
    train_set = []
    eval_set = []

    # Read the file vocalset_paths_labels.txt
    file_path = os.path.join(directory_path, "vocalset_paths_labels.txt")
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Split the lines into paths and labels
    data = [line.strip().split("|") for line in lines]

    # Shuffle the data randomly
    random.shuffle(data)

    # Calculate the number of data points to keep in the eval set
    eval_count = int(len(data) * eval_fraction)

    # Split the data into train and eval sets
    eval_set = data[:eval_count]
    train_set = data[eval_count:]

    # Write train_set to train_list.txt
    train_file_path = os.path.join(directory_path, "train_list.txt")
    with open(train_file_path, "w") as train_file:
        for path, label in train_set:
            train_file.write(f"{path}|{label}\n")

    # Write eval_set to val_list.txt
    eval_file_path = os.path.join(directory_path, "val_list.txt")
    with open(eval_file_path, "w") as eval_file:
        for path, label in eval_set:
            eval_file.write(f"{path}|{label}\n")



def main():
    dataset_root_dir = "../datasets/VocalSet/FULL"  # Replace with your desired root directory
    prepare_data(dataset_root_dir)
    process_vocalset("DataVocalSet")
    create_splits("DataVocalSet", eval_fraction=0.2)

if __name__ == "__main__":
    # main()
    create_splits("DataVocalSet", eval_fraction=0.2)

