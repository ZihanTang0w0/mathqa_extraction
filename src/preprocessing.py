import os
import json
from datasets import load_dataset
import random
from abc import ABC
from typing import List, Union
from transformers import AutoTokenizer
from model_utils import get_model, get_model_path
import re

class PreProcessor(ABC):
    def __init__(self, tokenizer_path=None):
        self.tokenizer = None
        if tokenizer_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
    def truncate_by_letter(self, text: str, max_truncate_len: int=2048*8):
        """
        First truncate lines that exceed `max_truncate_length`, then add row numbers to the beginning of each line in the given text.

        Args:
            text (str): text in str 

        Returns:
            str: The text with row numbers added to the beginning of each line.
        """
        
        # assuming lines are splitted by one of the following patterns: double spaces or \n
        lines = text.replace('  ', '\n').split('\n')
        
        line_truncates = []
        for line in lines:
            truncated_line = [line[i:i+max_truncate_len] for i in range(0, len(line), max_truncate_len)]
            line_truncates.extend(truncated_line)
        
        # combine lines by `max_truncate_len`
        combined_lines = []
        current_combination = ""
        max_combine_len = max(max([len(line) for line in line_truncates]), max_truncate_len)
        for line in line_truncates:
            if len(current_combination) + len(line) + (1 if current_combination else 0) > max_combine_len:
                combined_lines.append(current_combination)
                current_combination = line
            else:
                if current_combination:
                    current_combination += "\n"
                current_combination += line
        
        # add last combination if there is anything left
        if current_combination:
            combined_lines.append(current_combination)
    
        return combined_lines
    
    def truncate_by_token(self, text: str, max_truncate_len: int=2048) -> List[str]:
        # strip all the spaces before and after \n, assuming lines are splitted by one of the following patterns: double spaces or \n
        text.replace('  ', '\n')
        text = re.sub(r'\s*\n\s*', '\n', text)
        
        text_ids = self.tokenizer(
            text,
            truncation=True,
            max_length=max_truncate_len
        )
        decoded_text_lst = self.tokenizer.decode(text_ids["input_ids"])
        print(decoded_text_lst)

        
        return decoded_text_lst
    
    def add_row_numbers(self, text: str):
        lines = text.replace('  ', '\n').split('\n')
        numbered_lines = [f"[{i+1}] {line.strip()}" for i, line in enumerate(lines)]
        return '\n'.join(numbered_lines)


    def process_dataset(self, dataset, max_truncate_len: int=2048, mode="by_letter") -> List[str]:
        res = []
        for example in dataset:
            text = example["text"]
            if mode == "by_letter":
                truncates = self.truncate_by_letter(text, max_truncate_len=max_truncate_len)
            elif mode == "by_token":
                truncates = self.truncate_by_token(text, max_truncate_len=max_truncate_len)
            else:
                raise ValueError("Invalid truncate mode")
            # add row numbers
            truncates = [self.add_row_numbers(text) for text in truncates]
            res.extend(truncates)
        return res
    
    
if __name__ == '__main__':
    # test add_row_numbers on random open-web-math text
    owmath_homedir = '/data2/zhtang/from_hf/open_web_math'
    data_filelist = [os.path.join(owmath_homedir, shard) for shard in os.listdir(owmath_homedir)]
    owmath_ds = load_dataset('parquet', data_files=data_filelist[0])
    test_ds = owmath_ds['train'].select(range(10))
    
    # get tokenizer
    model_name = "Llama-3.1-70B-Instruct"
    preprocessor = PreProcessor(get_model_path(model_name))
    
    processed_dataset = preprocessor.process_dataset(test_ds, max_truncate_len=10, mode="by_token")
    print(processed_dataset[0])
    from utils import *
    write_to_jsonl(processed_dataset, "/home/zhtang/mathqa_extraction/output/test.txt")
    