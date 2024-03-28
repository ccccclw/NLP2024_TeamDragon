import argparse
import pickle
import operator
import collections
from collections import Counter
import argparse

# Function to preprocess a line by stripping whitespace and splitting into words
def preprocess_line(line):
    return line.strip().split()

# Function to calculate the coverage of a vocabulary for a given set of counts
def calculate_coverage(vocab, counts):
    total_count = sum(counts.values())
    covered_count = sum(counts.get(word, 0) for word in vocab)
    coverage = covered_count / total_count
    return coverage

def create_dictionary(file_name, lim=0):

    global_counter = Counter()

    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            words = line.strip().split()
            global_counter.update(words)

    combined_counter = global_counter

    if lim <= 4:
        lim = len(combined_counter) + 4

    vocab_count = combined_counter.most_common(lim - 4)
    total_counts = sum(combined_counter.values())
    coverage = 100.0 * sum(count for _, count in vocab_count) / total_counts
    print(f"Vocabulary coverage: {coverage:.2f}%")

    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<sos>": 2,
        "<eos>": 3,
    }

    for i, (word, count) in enumerate(vocab_count, start=4):
        vocab[word] = i

    return vocab


def parse_arguments():
    """
    Parse command-line arguments for building a vocabulary.

    Returns:
        argparse.Namespace: A namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Build vocabulary")

    parser.add_argument("--corpus", required=True, help="Path to the corpus file")
    parser.add_argument("--output", required=True, help="Path to the output file")
    parser.add_argument("--limit", type=int, default=0, help="Limit the vocabulary size (0 for no limit)")
    parser.add_argument("--char_mode", action="store_true", help="Treat input as characters instead of words")
    parser.add_argument("--sort_alpha", action="store_true", help="Sort the vocabulary alphabetically")
    parser.add_argument("--add_token", type=str, help="Add a custom token to the vocabulary")
    parser.add_argument("--groundhog_compat", action="store_true", help="Make the vocabulary compatible with Groundhog format")

    return parser.parse_args()

def construct_word(args):
    """
    Build a vocabulary based on the provided command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments parsed by `parse_arguments`.
    """
    if args.char_mode:
        word_counts = count_chars(args.corpus)
    else:
        word_counts = count_words(args.corpus)

    custom_tokens = args.add_token.split(";") if args.add_token else []
    vocab = remove_special_tokens(word_counts, custom_tokens)
    vocab = sort_by_freq(vocab)
    vocab = insert_tokens(vocab, custom_tokens)

    if args.limit != 0:
        vocab = vocab[:args.limit]
        print(f"Vocabulary coverage: {calculate_coverage(vocab, word_counts) * 100:.2f}%")

    if args.sort_alpha:
        token_count = len(custom_tokens)
        vocab = custom_tokens + sorted(vocab[token_count:])

    save_vocab(args.output, vocab)

def count_words(file_name):
    # Using a defaultdict to initialize counts to 0 by default
    word_counts = collections.defaultdict(int)

    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            words = preprocess_line(line)
            for word in words:
                word_counts[word] += 1

    return word_counts


def count_chars(file_name):
    # Using a defaultdict to initialize counts to 0 by default
    char_counts = collections.defaultdict(int)

    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            words = preprocess_line(line)
            for word in words:
                for char in word:
                    char_counts[char] += 1

    return char_counts


# Function to sort a vocabulary by frequency (descending order)
def sort_by_freq(vocab):
    # Convert the dictionary to a sorted list of (word, count) tuples
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
    # Return a list of words sorted by frequency
    return [word for word, count in sorted_vocab]

# Function to sort a vocabulary alphabetically
def sort_by_alpha(vocab):
    # Sort the keys (words) in the vocabulary alphabetically
    sorted_vocab = sorted(vocab.keys())
    return sorted_vocab


def save_vocab(file_name, vocab):
    # Create a new dictionary with words as keys and indices as values
    vocab_indices = {word: idx for idx, word in enumerate(vocab)}

    with open(file_name, "wb") as file:
        # Dump the vocabulary dictionary to the file using pickle
        pickle.dump(vocab_indices, file, pickle.HIGHEST_PROTOCOL)

# Function to parse tokens from a string
def parse_tokens(s):
    # Split the string by the semicolon (';') and return a list of tokens
    token_list = s.split(";")
    return token_list

# Function to remove special tokens from a vocabulary
def remove_special_tokens(vocab, tokens):
    # Create a new vocabulary dictionary without the special tokens
    new_vocab = {word: count for word, count in vocab.items() if word not in tokens}
    return new_vocab

# Function to insert tokens at the beginning of a vocabulary
def insert_tokens(vocab, tokens):
    # Create a new list with tokens inserted at the beginning
    new_vocab = tokens + vocab
    return new_vocab


def main():
    """
    Entry point of the script.

    This function parses command-line arguments, builds the vocabulary based on the arguments,
    and saves the vocabulary to a file using either the Groundhog format (pickle) or a custom format.
    """
    args = parse_arguments()

    if args.groundhog_compat:
        vocab = create_dictionary(args.corpus, args.limit)
        with open(args.output, "wb") as output_file:
            pickle.dump(vocab, output_file, pickle.HIGHEST_PROTOCOL)
    else:
        construct_word(args)

if __name__ == "__main__":
    main()
