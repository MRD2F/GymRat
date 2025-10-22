from pathlib import Path
import sys
sys.path.append(str(Path('../python').resolve()))
from create_clean_chunks import *
from transformers import pipeline, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
from transformers import AutoTokenizer

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def get_tokenizer(model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    #eos -> end of string token is the pad token
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def llama_token_len(text):
    return len(get_tokenizer().encode(text))

file="../data/books/Rebuilding Milo The Lifters Guide to Fixing Common Injuries and Building a Strong Foundation for Enhancing Performance (Dr. Aaron Horschig, Kevin Sonthana) (Z-Library).pdf"
chunk_size=250
chunk_overlap=30
# length_function=llama_token_len
# cc = CreateChunks(file, chunk_size,chunk_overlap, length_function )
# print('nevermid')
# llama_chunks=cc.get_clean_chunks()
with open("llama_chunks_text_chuck_size250_overlap30.json", "r") as f:
    llama_chunks = json.load(f)

print(f"Total chunks created: {len(llama_chunks)}")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # You can also use "fp4"
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quant_config,
    trust_remote_code=True
)

qa_gen = pipeline("text-generation", model=model, tokenizer=get_tokenizer(),  max_new_tokens=256)

def get_prompt_v2(text):
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
    tokenizer = get_tokenizer()
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



batch_size = 16
from tqdm import tqdm

text_chunks = llama_chunks
raw_outputs = []
n_repetitions = 3
samples = text_chunks[:]

for i in tqdm(range(0, len(samples), batch_size)):
    print(f"i: {i}")
    batch = samples[i:i + batch_size]
    print(f"Processing batch with samples {i, i + batch_size} ")

    for _ in range(n_repetitions):  # Repeat generation 3 times per batch
        prompt = [get_prompt_v2(chunk) for chunk in batch]
        
        raw_output = qa_gen(
            prompt, 
            max_new_tokens=256, 
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        raw_outputs.extend([o[0]["generated_text"] for o in raw_output])


import json

# Save to a file
with open(f"raw_outputs_samples{len(samples)}_nreps{n_repetitions}_new_tok{256}_chuck_size{250}_overlap{30}.json", "w") as f:
    json.dump(raw_outputs, f)


################################### CLEAN AFTER GENERATED TEXT

def escape_nested_quotes(json_str):
    def fix_quotes(match):
        key = match.group(1)
        value = match.group(2)
        # Escape unescaped quotes inside the string value
        value_fixed = re.sub(r'(?<!\\)"', r'\"', value)
        return f'"{key}": "{value_fixed}"'

    pattern = r'"(instruction|output)":\s*"((?:[^"\\]|\\.)*?)"(?=,\s*"|"\s*})'
    return re.sub(pattern, fix_quotes, json_str)
def check_for_null(instruction, output):
    if ("NULL" in instruction) | ("NULL" in output) :
        null_instruction = '"instruction": "NULL?.",'
        null_output = '"output": "NULL."'
        return null_instruction, null_output
    else:
        return instruction, output    

def merge_outputs(out):
    merged_out = '"output": "'
    out = out.replace('"instruction":', '"output":')
    for n in range(len(out.split("output"))):
        text = out.split("output")[n]
        #print(n, text, '\n')
        text = text[text.find('": "'):].replace('",', '').replace('"', '').replace(':', '')
        #print('*******',text)
        if ("}" in text): 
            text = text[:text.find('}')]
        elif ("{" in text):
            text = text[:text.find('{')]
        
        merged_out+=text
    merged_out = merged_out.strip()

    merged_out = merged_out+'.' if not merged_out.endswith('.') else merged_out

    merged_out = merged_out+'"' if not merged_out.endswith('"') else merged_out
    
    return merged_out.strip()
def is_malformed_json_like(text):
    # Check for empty keys like "": "value"
    # if re.search(r'"\s*"\s*:', text):
    #     return True
    
    # Check for unclosed braces
    if text.count('{') != text.count('}'):
        return True

    # Check for trailing commas before a closing brace
    if re.search(r',\s*}', text):
        return True

    # Check for improperly escaped quotes inside values
    if re.search(r':\s*"[^"]*"[^\s,}]', text):
        return True

    return False

def clean_and_merge_malformed_json(raw_text):
    # Step 1: Fix missing keys like '""' and replace them with 'label' or 'title'
    fixed = re.sub(r'"":', '"label":', raw_text)

    # Step 2: Split the entries if needed
    chunks = re.split(r'}\s*,\s*{', fixed)
    
    clean_chunks = []
    for chunk in chunks:
        chunk = chunk.strip().strip(',')  # Remove leading/trailing commas
        if not chunk.startswith('{'):
            chunk = '{' + chunk
        if not chunk.endswith('}'):
            chunk += '}'

        # Optional: validate minimal JSON structure before appending
        try:
            json.loads(chunk)
            clean_chunks.append(chunk)
        except json.JSONDecodeError:
            pass  # skip bad/incomplete JSON parts

    # Step 3: Wrap all valid chunks in a JSON array
    json_array_str = "[" + ", ".join(clean_chunks) + "]"

    # Step 4: Try to parse it
    try:
        return json.loads(json_array_str)
    except json.JSONDecodeError as e:
        print("Still not valid JSON:", e)
        return None

def remove_internal_double_virgolette(i, label):
    content = i.split(f'"{label}": ')[-1]
    if '"' in content[1:-1]:
        content = content[1:-1].replace('"', '')
        new = f'"{label}": ' + '"'+content+'"'
        if label == "instruction":
            new = new + ','
        return new
    else:
        return i
    
def instruction_output(output):
    raw_output = output.split("<|assistant|>")[1]
    raw_output = re.sub(r'"question":', '"instruction":', raw_output, flags=re.IGNORECASE)
    raw_output = re.sub(r'"answer":', '"output":', raw_output, flags=re.IGNORECASE)

    instruction = raw_output[raw_output.lower().find('"instruction":') : raw_output.find('",')]
    output = raw_output[raw_output.lower().find('"output":') : raw_output.find('."\n}')]
    output+='."'

    instruction = instruction.replace('"Instruction":', '"instruction":')
    output = output.replace('"Output":', '"output":')

    if "output" in instruction:
        instruction = instruction[:instruction.find("output")].replace('."\n', '').replace('"\n  "', '"')

    if '"instruction":' in output:
        output = output[output.find('"instruction":') : output.find('",')]
        output = output.replace('"instruction":', '')

    #Remove final dot, to add later the ?.,
    instruction = instruction.strip()[:-1] if instruction.strip().endswith(".") else instruction.strip()
    instruction = instruction if instruction.endswith("?") else instruction+"?"
    instruction = instruction+'.",'
    #print(f"INS: {instruction}")

    #return pre-defined null template if there "NULL on eather the insptructions or outputs"
    instruction, output = check_for_null(instruction, output)

    if "instruction" not in instruction:
        instruction = '"instruction": "NULL?.",'
        output = '"output": "NULL."'
    elif "output" not in output:
        output = '"output": "NULL".'
        instruction = '"instruction": "NULL?.",'

    if len(output.split("output")) > 2:
        output = merge_outputs(output)

    if "JSON object" in output:
        output = '"output": "NULL."'

    output = output.replace('"\n."', '."').replace('",\n', '')
    output = output.replace('"\n', '"').replace('`', '').replace('{', '').replace('}', '').replace('"\n', '"').replace('".', '."').strip()

    if ('{' in output):
        output = output[:output.find('}')]
    elif ('}' in output):
        output = output[:output.find('{')]

    output = remove_internal_double_virgolette(output, 'output')
    instruction = remove_internal_double_virgolette(instruction, 'instruction')
    return instruction, output, raw_output

raw_outputs = loaded_list 


#######SAVE INTO JSON
chunks_to_repete=[]
N_MAX = None
all_outputs, str_json = [], []
for n in range(len(raw_outputs) if N_MAX == None else N_MAX):
    instruction, output, _  = instruction_output(raw_outputs[n])
    output_results=output.split('"output":')[-1]

    if (len(output_results) < 15) | ("NULL" in output) | ("NULL" in instruction) | ("https//www.youtube" in output):
            instruction = '"instruction": "NULL.",'
            output = '"output": "NULL."'
            chunks_to_repete.append(n)

    output = output.replace('"Question:', '')
    instruction = instruction.replace('?"?."', '?."').replace('"?."', '').replace('"?.",', ',').replace('- "?."', ',')
    if output.endswith('""'):
        output = output.replace('""', '"')

    json_str = "{" + instruction + output + "}"
    str_json.append(json_str)

    try:
        parsed = json.loads(json_str.replace("\n", "").replace('"}``."}', '"}').replace('.""', '."'))        
        all_outputs.append(parsed)

    except json.JSONDecodeError as e:
        print("********************************")
        print(f"[!] JSON decode error at chunk {n}: {e}")
        print(f"Error input : {instruction}")
        print(f"Error output: {output}")
        print('json_str: ' ,json_str)
        print("********************************")


############### CLEAN JSON ############àà

import os
import pandas as pd
import numpy as np
from pprint import pprint

if False:
    json_file_name = "qa_outputs_800_batch16_4bit_30072025.json"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)

existing_data = all_outputs
df = pd.DataFrame(existing_data)


# Cleaning/filter function
def is_valid(text):
    if pd.isna(text):  
        return False
    if not isinstance(text, str):  # Check if not a string
        return False

    text = text.strip().lower()
    # Check for known invalid values or if 'null' is present anywhere
    invalid_keywords =["null", "none", "nan", "n/a"] 
    if any(bad in text for bad in invalid_keywords):
        return False

    #minimum content length
    if len(text) < 5:  
        return False
    return True

# Apply filter to both columns
filtered_df = df[df['instruction'].apply(is_valid) & df['output'].apply(is_valid)]

filtered_df = filtered_df.drop_duplicates()

cleaned_data = filtered_df.to_dict(orient="records")
n_records=filtered_df.shape[0]

json_file_name= "raw_outputs_samples1260_nreps3_new_tok256_chuck_size250_overlap30"

with open(f"cleaned_{n_records}_{json_file_name}", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)