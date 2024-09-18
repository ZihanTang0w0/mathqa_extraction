from openai import OpenAI
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from models.base_model import BaseModel
import time

class DeepseekCode(BaseModel):
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    def generate_single(self, example):
        attempt_count = 0
        max_attempts = 5
        while attempt_count < max_attempts:
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-coder",
                    messages=[
                        {"role": "system", "content": "You are a highly skilled student proficient in solving math problems."},
                        {"role": "user", "content": self.generate_prompt(example)},
                    ],
                    stream=False,
                )
                return response.choices[0].message.content
            except Exception as e:
                attempt_count += 1
                print(f"Error occurred when calling openai api! {e}")
                time.sleep(15)
            return None
    