import argparse
import os
import json
from typing import List
from tqdm import tqdm
import transformers
import torch
from abc import ABC
from model_utils import get_model, get_model_path
from datasets import load_dataset
from preprocessing import preprocess_dataset
from utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

HOME_DIR = "/home/zhtang/mathqa_extraction"
SYSTEM_PROMPT = "You are an expert in identifying and extracting math question-answer pairs from unstructured text. Your task is to:\n1. Identify math questions in the provided text and extract them.\n2. Locate and extract their corresponding answers, if they exist in the text.\n3. If no answer is found, you should still extract the question.\n4. For each extracted question and answer, return the row number ranges where they are found in the text. Use the following format: Output Format: Question: Extracted question text\n- Found in rows: X to Y\nAnswer: Extracted answer text (or Answer not found if it does not exist)\n- Found in rows: A to B (or indicate if the answer is missing)"


class ExtractionPipeline(ABC):
    def __init__(
            self, 
            model_path=None, 
            output_path=None
        ):
        if model_path is not None:
            self.gen_pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_path,
                    model_kwargs={"torch_dtype": torch.float16},
                    device_map="auto"
                )
            self.gen_pipeline.tokenizer.pad_token_id = self.gen_pipeline.tokenizer.eos_token_id
            self.gen_pipeline.tokenizer.padding_side = 'left'
        self.output_path = output_path
    
    def pre_process(self, dataset, max_truncate_len=2048*8) -> List[str]:
        """
        Add row numbers to contents in the dataset and truncate them by `max_truncate_len`
        Return a list of truncated texts in string
        """
        return preprocess_dataset(dataset, max_truncate_len=max_truncate_len)
    
    def text_to_message(self, text):
        return [
                    {"role": "system", 
                        "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ]
    
    def batch_get_messages(self, text_lst) -> List[List]:
        return [self.text_to_message(text) for text in text_lst]
        
    def generate_single(self, text):
        messages = self.text_to_message(text)
        outputs = self.gen_pipeline(
                            messages,
                            max_new_tokens=2048,
                            do_sample=False,
                            )
        
        return outputs[0]["generated_text"][-1]['content']
    
    def generate_batch(self, dataset, batchsize=8):
        messages = self.batch_get_messages([example['text'] for example in dataset])
        return self.gen_pipeline(
                messages, 
                batch_size=batchsize,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=self.gen_pipeline.tokenizer.eos_token_id
                )
        
    def save_generated(self, generated, output_path):
        res = []
        for i, item in enumerate(generated): # item of format [{},..., {"generated_text": [{"role": system,}, {"role": "user"}, {"role": assistant, "content": ...}]}]
            text2qa_data = {}
            text2qa_data["id"] = i
            text2qa_data["text"] = generated["generated_text"][1]["content"]
            text2qa_data["preliminary_extracted_qa"] = generated["generated_text"][-1]["content"]
            res.append(text2qa_data)
        
        write_to_jsonl(res, output_path)
            
    def end2end_extract(self, dataset, batchsize=8, output_path=None):
        #preprocess dataset
        text_lst = preprocess_dataset(dataset)
        messages = self.batch_get_messages(text_lst)
        
        #batch generate
        generated = self.gen_pipeline(
            messages,
            batch_size=batchsize,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=self.gen_pipeline.tokenizer.eos_token_id
        )
        
        #save outputs
        if output_path is None:
            if self.output_path is None:
                output_path = f"{HOME_DIR}/output_extracted_qa_data.jsonl"
            else:
                output_path = self.output_path
        self.save_generated(generated, output_path)
        
if __name__ == "__main__":
    model_name = "Llama-3.1-70B-Instruct"
    model_path = get_model_path(model_name)
    
    # open-web-math dataset
    owmath_homedir = '/data2/zhtang/from_hf/open_web_math'
    assert(os.path.exists(owmath_homedir))
    data_filelist = [os.path.join(owmath_homedir, shard) for shard in os.listdir(owmath_homedir)]
    
    # print(data_filelist[0])
    owmath_ds = load_dataset('parquet', data_files=data_filelist[0], split="train")
    
    # test dataset
    test_ds = owmath_ds.select(range(0, 2))

    # pipe
    pipeline = ExtractionPipeline(
        model_path=None,
        output_path=f"{HOME_DIR}/output/output_extracted_qa_data_1.jsonl"
    )
    
    processed_data = pipeline.pre_process(test_ds)
    md_data = [convert_to_markdown(text) for text in processed_data]
    write_to_jsonl(processed_data, pipeline.output_path)
    
    # pipeline.end2end_extract(test_ds, batchsize=2)