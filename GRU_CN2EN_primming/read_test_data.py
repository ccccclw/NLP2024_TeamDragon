import os
import math
import random

def read_file(file_path):
#Read a text file and return the content as a list of lines.
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.readlines()
    return data

def shuffle_indices(length, sample_size=0.01):
    """
    Generate a shuffled list of indices for sampling.

    Args:
        length (int): Total length of the data.
        sample_size (float, optional): Fraction of data to sample. Default is 0.01.

    Returns:
        list: A shuffled list of indices for sampling.
    """
    sample_count = max(1, int(length * sample_size))
    indices = list(range(length))
    random.shuffle(indices)
    return indices[:sample_count]

def generate_test_data(src_file, tgt_file, output_dir="data"):
    """
    Generate test data from source and target files by sampling a fraction of the data.

    Args:
        src_file (str): Name of the source file.
        tgt_file (str): Name of the target file.
        output_dir (str, optional): Directory to save the test files. Default is "data".

    Returns:
        None
    """
    src_path = os.path.join(output_dir, src_file)
    tgt_path = os.path.join(output_dir, tgt_file)

    src_data = read_file(src_path)
    tgt_data = read_file(tgt_path)

    assert len(src_data) == len(tgt_data), "Source and target files must have the same number of lines."

    sample_indices = shuffle_indices(len(src_data), sample_size=0.01)

    src_test = [src_data[idx] for idx in sample_indices]
    tgt_test = [tgt_data[idx] for idx in sample_indices]

    src_base, _ = os.path.splitext(src_file)
    tgt_base, _ = os.path.splitext(tgt_file)

    src_test_path = os.path.join(output_dir, f"{src_base}.test.txt")
    tgt_test_path = os.path.join(output_dir, f"{tgt_base}.test.txt")

    with open(src_test_path, "w", encoding="utf-8") as src_file, open(tgt_test_path, "w", encoding="utf-8") as tgt_file:
        src_file.writelines(src_test)
        tgt_file.writelines(tgt_test)

if __name__ == "__main__":
    generate_test_data("cn.txt", "en.txt")