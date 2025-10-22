# from pathlib import Path
# import sys
# sys.path.append(str(Path('../python').resolve()))
#from create_clean_chunks import *

import json
from tqdm import tqdm
#Not used for now, didn't seem to improve much the results, and needed to be adapted to the next classes to process the raw outputs
from generated_prompt import prompt_template 
from transformers import pipeline, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

# import os
# import pandas as pd

## Clean chunk text from a .pdf file is created using the CreateChunks class from the create_clean_chunks.py script
## In this script the chunked cleaned text is used as input of a LLM to generate a QA list
## This list can be saved, or used directly, in the CreateJsonQA class of the create_json_qa.py script,
## which provides the final input for the LLM fine-tunning

class GenerateQAContent:
    def __init__(self, file_name, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", compute_chunks=False):
        self.file_name = file_name
        self.model_id = model_id

    def get_text(self):
        # with open(self.file_name, "r") as f:
        #     llama_chunks = json.load(f)
        with open(self.file_name, 'r') as f:
            llama_chunks = f.readlines()
        return llama_chunks 
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left", 
        #                                           max_tokens=256)
        #eos -> end of string token is the pad token
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def llama_token_len(self, text):
        return len(self.get_tokenizer().encode(text))

    def get_transformers_pipeline(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # You can also use "fp4"
            bnb_4bit_compute_dtype="float16",
            llm_int8_enable_fp32_cpu_offload=True   # ✅ allow safe CPU offload 
            #By default, when offloading, Hugging Face expects those CPU modules to run in full 32-bit precision (FP32). Since you didn’t explicitly tell it how to handle that case, it raises the error:
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True
        )

        qa_gen = pipeline("text-generation", model=model, tokenizer=self.get_tokenizer(), 
                           max_new_tokens=512)
        return qa_gen
    
    def get_prompt(self, text):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise and helpful medical tutor. "
                    "Based on the provided text, generate a JSON object with exactly ONE question (as 'instruction') and ONE answer (as 'output').\n\n"
                    "- The content must relate to health, exercise, sports, fitness, or physiotherapy.\n"
                    "- Do not include multiple questions or answers.\n"
                    "- Do not repeat the instruction in the output.\n"
                    "- Keep the output brief and informative.\n"
                    "- If the text is not relevant, return: {\"instruction\": \"NULL\", \"output\": \"NULL\"}\n\n"
                    "- Respond ONLY with the JSON object. Do NOT include any explanation or commentary."
                ),
            },
            {
                "role": "user",
                "content": text.strip()
            },
        ]
        tokenizer = self.get_tokenizer()
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_content(self, n_chunks_intervals=None, n_repetitions=3, save_json=False, batch_size=16):
        text_chunks = self.get_text()
        raw_outputs = []
        samples = text_chunks[:] if n_chunks_intervals == None else text_chunks[n_chunks_intervals[0]:n_chunks_intervals[1]]


        qa_gen = self.get_transformers_pipeline()
        for i in tqdm(range(0, len(samples), batch_size)):
            print(f"i: {i}")
            batch = samples[i:i + batch_size]
            print(f"Processing batch with samples {i, i + batch_size} ")

            for _ in range(n_repetitions):  # Repeat generation 3 times per batch
                #prompt = [prompt_template(chunk,1) for chunk in batch]
                prompt = [self.get_prompt(chunk) for chunk in batch]
                
                raw_output = qa_gen(
                    prompt, 
                    max_new_tokens=512, 
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95
                )
                
                raw_outputs.extend([o[0]["generated_text"] for o in raw_output])
        
        if save_json:
            json_output_name= f"../data/raw_outputs_samples{len(samples)}_nreps{n_repetitions}_metallama_maxtoken{128}max_new_tokens{512}"
            with open(f"{json_output_name}.json", "w") as f:
                json.dump(raw_outputs, f)
        
        return raw_outputs
    
test=False
if test: 
    file_name = "../data/rebuilding_milo_chunks_docling_max_tokens128_min_tokens50_meta_llama3p18B.txt" 
    model_id = "meta-llama/Llama-3.1-8B-Instruct" #TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    save_json=False
 
    #For all chunks set to None
    n_chunks_intervals=[30,31] 
    n_repetitions = 1

    gc = GenerateQAContent(file_name, model_id)
    text = gc.get_text()
    raw_outputs = gc.generate_content(n_chunks_intervals=n_chunks_intervals, n_repetitions=n_repetitions, 
                                      save_json=save_json, batch_size=16)
    #print(raw_outputs)