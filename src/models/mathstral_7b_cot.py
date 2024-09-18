import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import transformers
import torch
from models.base_model import BaseModel


class Mathstral_7b_cot(BaseModel):
    def __init__(self, model_path=None):
        self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_path,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
        
    def generate_single(self, example):
        messages = [
                    {"role": "system", "content": "You are a highly skilled student proficient in solving math problems."},
                    {"role": "user", "content": self.generate_prompt(example)},
                ]

        outputs = self.pipeline(
                            messages,
                            max_new_tokens=2048,
                            do_sample=False,
                            pad_token_id=self.pipeline.tokenizer.eos_token_id
                            )
        return outputs[0]["generated_text"][-1]['content']
