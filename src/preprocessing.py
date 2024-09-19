import os
import json
from datasets import load_dataset
import random
from abc import ABC
from typing import List, Union
from transformers import AutoTokenizer
from model_utils import get_model, get_model_path
from tqdm import tqdm
import re

class PreProcessor(ABC):
    def __init__(self, tokenizer_path=None):
        """
        Args:
            tokenizer_path (str, optional): path to the tokenizer, if given, will load the tokenizer, otherwise will not load a tokenizer. Defaults to None.
        """
        self.tokenizer = None
        if tokenizer_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
    def segment_by_letter(self, text: str, max_segment_len: int=2048*8):
        """
        First segment lines that exceed `max_segment_length`, then add row numbers to the beginning of each line in the given text.

        Args:
            text (str): text in str 

        Returns:
            str: The text with row numbers added to the beginning of each line.
        """
        
        # assuming lines are splitted by one of the following patterns: double spaces or \n
        lines = text.replace('  ', '\n').split('\n')
        
        line_segments = []
        for line in lines:
            segmentd_line = [line[i:i+max_segment_len] for i in range(0, len(line), max_segment_len)]
            line_segments.extend(segmentd_line)
        
        # combine lines by `max_segment_len`
        combined_lines = []
        current_combination = ""
        max_combine_len = max(max([len(line) for line in line_segments]), max_segment_len)
        for line in line_segments:
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
    
    def segment_by_token(self, text: str, max_segment_len: int=2048) -> List[str]:
        """
        Segment the given text by max_segment_len tokens.

        1. First strip all the spaces before and after \n, assuming lines are splitted by one of the following patterns: double spaces or \n.
        2. Segment each line by max_segment_len, then combine lines sequentially of which the lengths sum up to max_segment_len.
        3. Decode and return the segmented text.

        Args:
            text (str): text in str
            max_segment_len (int, optional): the maximum length of each segment. Defaults to 2048.

        Returns:
            List[str]: the segmented text.
        """
        # 1. strip all the spaces before and after \n, assuming lines are splitted by one of the following patterns: double spaces or \n
        text.replace('  ', '\n')
        text = re.sub(r'\s*\n\s*', '\n', text)
        lines = [splitted_text + '\n' for i, splitted_text in enumerate(text.split('\n'))]
        
        # text_ids: List[List[int]]
        text_ids = self.tokenizer(lines)["input_ids"]
        
        # 2. segment each line by max_segment_len, then combine lines sequentially of which the lengths sum up to max_segment_len
        # segment
        segmented_ids = []
        for i, line in enumerate(text_ids):
            line_segmented = [line[i:i+max_segment_len] for i in range(0, len(line), max_segment_len+1)]
            segmented_ids.extend(line_segmented)
        
        # combine
        combined_lines = []
        current_combination = []
        max_combine_len = max(max([len(line) for line in segmented_ids]), max_segment_len+1)
        
        for i, line in enumerate(segmented_ids):
            if len(line) + len(current_combination) + (1 if current_combination else 0) > max_combine_len:
                combined_lines.append(current_combination)
                current_combination = line
            else:
                if current_combination:
                    line = line[1:] # add \n to the beginning of the line
                current_combination.extend(line)
                
                    
        
        if current_combination:
            combined_lines.append(current_combination)
        
        # 3. decode and return
        return [self.tokenizer.decode(combined_line[1:]) for combined_line in combined_lines] 
    
    def add_row_numbers(self, text: str):
        """
        Add row numbers to the given text.

        The input text is first split by \n, then each line is striped and numbered with [1], [2], [3], ... respectively.
        The last line is ignored in the numbering process.

        Args:
            text (str): the input text

        Returns:
            str: the input text with row numbers added
        """
        lines = text.replace('  ', '\n').split('\n')
        numbered_lines = [f"[{i+1}] {line.strip()}" for i, line in enumerate(lines) if i != len(lines)-1]
        return '\n'.join(numbered_lines)

    def process_dataset(self, dataset, max_segment_len: int=2048, mode="by_token") -> List[List[str]]:
        """
        Process a dataset by segmenting each example's text into segments of at most `max_segment_len` length.
        The segmentation is done either by letter or by token, depending on the `mode` argument.
        
        Args:
            dataset (List[Dict[str, str]]): the input dataset
            max_segment_len (int): the maximum length of each segment
            mode (str): either "by_letter" or "by_token", indicating how to segment the text
        
        Returns:
            List[List[str]]: the processed dataset, where each inner list contains the segmented text
        """
        res = []
        for example in tqdm(dataset):
            text = example["text"]
            if mode == "by_letter":
                segments = self.segment_by_letter(text, max_segment_len=max_segment_len)
            elif mode == "by_token":
                segments = self.segment_by_token(text, max_segment_len=max_segment_len)
            else:
                raise ValueError("Invalid segment mode")
            
            # add row numbers
            segments = [self.add_row_numbers(text) for text in segments]
            res.append(segments)
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
    
    print(test_ds[0]['text'])
    processed_dataset = preprocessor.process_dataset(test_ds, max_segment_len=2048, mode="by_token")
    
    print(processed_dataset[0])
    from utils import *
    write_to_jsonl(processed_dataset, "/home/zhtang/mathqa_extraction/output/test.txt")
    
    