# -*- coding: utf-8 -*-

import nltk
# nltk.download('punkt')  

def tokenize_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    tokenized_lines = []
    for line in lines:
        tokens = nltk.word_tokenize(line.strip())
        tokenized_line = ' '.join(tokens)
        tokenized_lines.append(tokenized_line)

    with open(output_file, 'w', encoding='utf-8') as f:
        for tokenized_line in tokenized_lines:
            f.write(tokenized_line + '\n')

input_file = 'data/en.txt'
output_file = 'data/en_output.txt' 
tokenize_file(input_file, output_file)