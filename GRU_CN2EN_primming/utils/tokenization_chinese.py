# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 02:08:06 2024

@author: ALKAIM
"""

import jieba

def tokenize_chinese_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    tokenized_lines = []
    for line in lines:
        tokens = jieba.cut(line.strip())
        tokenized_line = ' '.join(tokens)
        tokenized_lines.append(tokenized_line)

    with open(output_file, 'w', encoding='utf-8') as f:
        for tokenized_line in tokenized_lines:
            f.write(tokenized_line + '\n')

input_file = 'data/cn.txt'  # 替换成你的输入文件路径
output_file = 'data/cn_output.txt'  # 替换成你的输出文件路径
tokenize_chinese_file(input_file, output_file)
