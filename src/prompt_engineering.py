import json
import os
from datasets import load_dataset


SYSTEM_PROMPT = "You are an expert in identifying and extracting math question-answer pairs from unstructured text. Your task is to:\n1. Identify math questions in the provided text and extract them.\n2. Locate and extract their corresponding answers, if they exist in the text.\n3. If no answer is found, you should still extract the question.\n4. For each extracted question and answer, return the row number ranges where they are found in the text. Use the following format: Output Format: Question: Extracted question text\n- Found in rows: X to Y\nAnswer: Extracted answer text (or Answer not found if it does not exist)\n- Found in rows: A to B (or indicate if the answer is missing)"

FEWSHOT_EXAMPLES = [
    {"text": "", "question-answer pairs": ""},
    {"text": "", "question-answer pairs": ""},
    {"text": "", "question-answer pairs": ""},
    {"text": "", "question-answer pairs": ""},
    {"text": "", "question-answer pairs": ""},
                ] # TODO preferably 5 examples


def text_to_message(text: str):
    sys_content = SYSTEM_PROMPT + "\n" + "examples:\n" + json.dumps(FEWSHOT_EXAMPLES, indent=4)
    return [
                {"role": "system", 
                    "content": sys_content},
                {"role": "user", "content": text},
            ]

if __name__ == "__main__":
    # open-web-math dataset
    owmath_homedir = '/data2/zhtang/from_hf/open_web_math'
    assert(os.path.exists(owmath_homedir))
    data_filelist = [os.path.join(owmath_homedir, shard) for shard in os.listdir(owmath_homedir)]

    owmath_ds = load_dataset('parquet', data_files=data_filelist[0], split="train")
    text = owmath_ds[0]['text']
    
    print(str(text_to_message(text)))
    