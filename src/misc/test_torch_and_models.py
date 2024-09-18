import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os


MODEL_LIST = [
"Qwen2-Math-72B-Instruct",
"Mathstral-7B-v0.1",
"NuminaMath-7B-CoT",
"DeepSeek-Coder-V2-Instruct-0724",
"DeepSeek-V2.5",
"Llama-3.1-70B-Instruct",
]

def test_torch():
    if torch.cuda.is_available():
            # Print all CUDA devices
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i))
    else:
        print("CUDA is not available. No GPU device found.")
        
    # 0-7 are avail device ids
    tensor_a = torch.tensor([1, 2, 3], device='cuda:1')

    device = torch.device('cuda:7')  
    tensor_b = torch.tensor([1, 2, 3], device=device)

    print(tensor_a, tensor_b)

def test_model_eval(prompt: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    device = torch.device('cuda:7')
    device_map = {"":7}
    
    # quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model_name = "Qwen2-72B-Instruct"
    model_dir = f'/data2/ckpts/{model_name}'
    assert(os.path.exists(model_dir))
    
    print(f'loading {model_name} model...')
    model = AutoModelForCausalLM.from_pretrained(model_dir, 
                                                 device_map=device_map,
                                                 torch_dtype=torch.float16,
                                                 quantization_config=bnb_config).eval()
    
    print(f'loading {model_name} tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    print(f'generating...')
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=2048
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    
def test_tokenizer(model_path, text):
    """
    tokenizer(text) -> {"input_ids": List[tokens], "attention_mask": List[attentions]}
    
    tokenizer(List[text]) -> {"input_ids": List[List[tokens]], "attention_mask": List[List[attentions]]}
    
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # print(tokenizer(text))
    
    print(f'bos: {tokenizer("")}')
    backslash_n = tokenizer("\n")
    print(f'bos and backslash-n" {backslash_n}')
    
    row_number_text = "[1] hello\n [2] world"
    print(f"row number: {tokenizer(row_number_text)}")
    
    

if __name__ == '__main__':
    text = "[1] # “Broken” curve with ContourPlot [closed]\n[2] I am trying to plot a contour ellipse\n[3] ContourPlot[2.8034950798347694*10^8 + 346.6795282575007 x^2 -\n[4] 929387.1591151787 y + 770.3723456790123 y^2 +\n[5] x (-623273.1898861742 + 1032.7522877535007 y) == 50000, {x, -600,\n[6] 600}, {y, 0, 650}]\n[7] But the ellipse appears to be \"broken\", is there any remedy for this? On doing the same thing on WolframAlpha I got a nice smooth ellipse though.\n[8] ## closed as off-topic by MarcoB, m_goldberg, Bob Hanlon, Öskå, C. E.Sep 5 '15 at 12:21\n[9] This question appears to be off-topic. The users who voted to close gave this specific reason:\n[10] • \"This question arises due to a simple mistake such as a trivial syntax error, incorrect capitalization, spelling mistake, or other typographical error and is unlikely to help any future visitors, or else it is easily found in the documentation.\" – MarcoB, m_goldberg, Bob Hanlon, Öskå, C. E.\n[11] If this question can be reworded to fit the rules in the help center, please edit the question.\n[12] ContourPlot["
    text_lst = [text, text]
    
    model_path = "/data2/ckpts/Meta-Llama-3.1-70B-Instruct"
    
    test_tokenizer(model_path, text)
    
    # print("\n\n\n\n-------------------")
    
    # test_tokenizer(model_path, text_lst) 

    
    
    
    