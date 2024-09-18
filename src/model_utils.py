import os
from tqdm import tqdm
from models.base_model import BaseModel

MODEL_LIST = [
"Qwen2-Math-72B-Instruct",
"Mathstral-7B-v0.1",
"NuminaMath-7B-CoT",
"DeepSeek-Coder-V2-Instruct-0724", 
"DeepSeek-V2.5",
"Llama-3.1-70B-Instruct",
]

def get_model(model_name, args) -> BaseModel:
    if model_name == "Qwen2-Math-72B-Instruct":
        pass
    
    elif model_name == "Llama-3.1-70B-Instruct":
        model_path = "/data2/ckpts/Meta-Llama-3.1-70B-Instruct"
        if not os.path.exists(model_path):
            raise ValueError(f'Invalid model path {model_path}')
        from models.llama3_1_70b_instruct import LLaMa3_1_70b_instruct
        return LLaMa3_1_70b_instruct(model_path=model_path)
    
    elif model_name == "Mathstral-7B-v0.1":
        model_path = "/data2/zhuang/OlympicArena/OlympicArena/code/ckp/mathstral-7B-v0.1"
        if not os.path.exists(model_path):
            raise ValueError(f'Invalid model path {model_path}')
        from models.mathstral_7b_cot import Mathstral_7b_cot
        return Mathstral_7b_cot(model_path=model_path)
    
    elif model_name == "NuminaMath-7B-CoT":
        model_path = "/data2/ckpts/NuminaMath-7B-CoT"
        if not os.path.exists(model_path):
            raise ValueError(f'Invalid model path {model_path}')
        from models.numinamath_7b_cot import Numinamath_7b_cot
        return Numinamath_7b_cot(model_path=model_path)
    
    elif model_name == "DeepSeek-Coder-V2-Instruct-0724":
        from models.deepseek_coder import DeepseekCode
        return DeepseekCode(args.api_key)
    
    elif model_name == "DeepSeek-V2.5":
        from models.deepseek_v25 import Deepseek 
        return Deepseek(args.api_key)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
def get_model_path(model_name, args=None) -> str: 
    if model_name == "Qwen2-Math-72B-Instruct":
        pass
    
    elif model_name == "Llama-3.1-70B-Instruct":
        model_path = "/data2/ckpts/Meta-Llama-3.1-70B-Instruct"
        if not os.path.exists(model_path):
            raise ValueError(f'Invalid model path {model_path}')
        
        return model_path
    
    elif model_name == "Mathstral-7B-v0.1":
        model_path = "/data2/zhuang/OlympicArena/OlympicArena/code/ckp/mathstral-7B-v0.1"
        if not os.path.exists(model_path):
            raise ValueError(f'Invalid model path {model_path}')
        
        return model_path
    
    elif model_name == "NuminaMath-7B-CoT":
        model_path = "/data2/ckpts/NuminaMath-7B-CoT"
        if not os.path.exists(model_path):
            raise ValueError(f'Invalid model path {model_path}')
        
        return model_path
    
    else:
        raise ValueError(f"Unknown model: {model_name}")