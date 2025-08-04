from pathlib import Path
import sys
sys.path.append(str(Path('../python').resolve()))
from create_clean_chunks import *
from transformers import pipeline, AutoModelForCausalLM, BitsAndBytesConfig
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import pandas as pd


class GenerateQAContent:
    def __init__(self, file_name, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", compute_chunks=False):
        self.file_name = file_name
        self.model_id = model_id

    def get_text(self):
        with open(self.file_name, "r") as f:
            llama_chunks = json.load(f)
        return llama_chunks 
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
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
            bnb_4bit_compute_dtype="float16"
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True
        )

        qa_gen = pipeline("text-generation", model=model, tokenizer=self.get_tokenizer(), 
                           max_new_tokens=256)
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

    def generate_content(self, n_max_chunks=None, n_repetitions=3, json_output_name=None, batch_size=16):
        text_chunks = self.get_text()
        raw_outputs = []
        samples = text_chunks[:] if n_max_chunks == None else text_chunks[:n_max_chunks]

        qa_gen = self.get_transformers_pipeline()
        for i in tqdm(range(0, len(samples), batch_size)):
            print(f"i: {i}")
            batch = samples[i:i + batch_size]
            print(f"Processing batch with samples {i, i + batch_size} ")

            for _ in range(n_repetitions):  # Repeat generation 3 times per batch
                prompt = [self.get_prompt(chunk) for chunk in batch]
                
                raw_output = qa_gen(
                    prompt, 
                    max_new_tokens=256, 
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95
                )
                
                raw_outputs.extend([o[0]["generated_text"] for o in raw_output])
        
        if json_output_name:
            #with open(f"raw_outputs_samples{len(samples)}_nreps{n_repetitions}_new_tok{256}_chuck_size{250}_overlap{30}.json", "w") as f:
            with open(f"{json_output_name}.json", "w") as f:
                json.dump(raw_outputs, f)
        
        return raw_output
test=True
if test:    
    file_name = "llama_chunks_text_chuck_size250_overlap30.json"
    gc = GenerateQAContent(file_name)
    text = gc.get_text()
    raw = gc.generate_content(n_max_chunks=3)
    print(raw)

# #####################################################################################
# #####################################################################################
# ####################### CLEANING RAW GENERATED CONTENT ##############################
# #####################################################################################
# #####################################################################################

# class CreateJsonQA:
#     def __init__(self, raw_content_name=None, input_text_file_name=None, n_max_chunks=None):
#         self.raw_content_name = raw_content_name
#         self.input_text_file_name = input_text_file_name
#         self.n_max_chunks = n_max_chunks
    
#     def get_raw_output(self):
#         if self.raw_content_name:
#             if os.path.exists(self.raw_content_name):
#                 with open(self.raw_content_name, "r", encoding="utf-8") as f:
#                     raw_outputs = json.load(f)
#             else:
#                 print(f"Input raw_content_name: {self.raw_content_name} provided to class CreateJsonQA, couldn't be opened.")

#         else:
#             gc = GenerateQAContent(self.input_text_file_name)
#             raw_outputs = gc.generate_content(self.n_max_chunks)
        
#         return raw_outputs

#     def escape_nested_quotes(self, json_str):
#         def fix_quotes(match):
#             key = match.group(1)
#             value = match.group(2)
#             # Escape unescaped quotes inside the string value
#             value_fixed = re.sub(r'(?<!\\)"', r'\"', value)
#             return f'"{key}": "{value_fixed}"'

#         pattern = r'"(instruction|output)":\s*"((?:[^"\\]|\\.)*?)"(?=,\s*"|"\s*})'
#         return re.sub(pattern, fix_quotes, json_str)
    
#     def check_for_null(self, instruction, output):
#         if ("NULL" in instruction) | ("NULL" in output) :
#             null_instruction = '"instruction": "NULL?.",'
#             null_output = '"output": "NULL."'
#             return null_instruction, null_output
#         else:
#             return instruction, output    

#     def merge_outputs(self, out):
#         merged_out = '"output": "'
#         out = out.replace('"instruction":', '"output":')
#         for n in range(len(out.split("output"))):
#             text = out.split("output")[n]
#             #print(n, text, '\n')
#             text = text[text.find('": "'):].replace('",', '').replace('"', '').replace(':', '')
#             #print('*******',text)
#             if ("}" in text): 
#                 text = text[:text.find('}')]
#             elif ("{" in text):
#                 text = text[:text.find('{')]
            
#             merged_out+=text
#         merged_out = merged_out.strip()

#         merged_out = merged_out+'.' if not merged_out.endswith('.') else merged_out

#         merged_out = merged_out+'"' if not merged_out.endswith('"') else merged_out
        
#         return merged_out.strip()
    
#     def is_malformed_json_like(self, text):
#         # Check for empty keys like "": "value"
#         # if re.search(r'"\s*"\s*:', text):
#         #     return True
        
#         # Check for unclosed braces
#         if text.count('{') != text.count('}'):
#             return True

#         # Check for trailing commas before a closing brace
#         if re.search(r',\s*}', text):
#             return True

#         # Check for improperly escaped quotes inside values
#         if re.search(r':\s*"[^"]*"[^\s,}]', text):
#             return True

#         return False

#     def clean_and_merge_malformed_json(self, raw_text):
#         # Step 1: Fix missing keys like '""' and replace them with 'label' or 'title'
#         fixed = re.sub(r'"":', '"label":', raw_text)

#         # Step 2: Split the entries if needed
#         chunks = re.split(r'}\s*,\s*{', fixed)
        
#         clean_chunks = []
#         for chunk in chunks:
#             chunk = chunk.strip().strip(',')  # Remove leading/trailing commas
#             if not chunk.startswith('{'):
#                 chunk = '{' + chunk
#             if not chunk.endswith('}'):
#                 chunk += '}'

#             # Optional: validate minimal JSON structure before appending
#             try:
#                 json.loads(chunk)
#                 clean_chunks.append(chunk)
#             except json.JSONDecodeError:
#                 pass  # skip bad/incomplete JSON parts

#         # Step 3: Wrap all valid chunks in a JSON array
#         json_array_str = "[" + ", ".join(clean_chunks) + "]"

#         # Step 4: Try to parse it
#         try:
#             return json.loads(json_array_str)
#         except json.JSONDecodeError as e:
#             print("Still not valid JSON:", e)
#             return None

#     def remove_internal_double_virgolette(self, i, label):
#         content = i.split(f'"{label}": ')[-1]
#         if '"' in content[1:-1]:
#             content = content[1:-1].replace('"', '')
#             new = f'"{label}": ' + '"'+content+'"'
#             if label == "instruction":
#                 new = new + ','
#             return new
#         else:
#             return i
        
#     def create_instruction_output(self, output):
#         raw_output = output.split("<|assistant|>")[1]
#         raw_output = re.sub(r'"question":', '"instruction":', raw_output, flags=re.IGNORECASE)
#         raw_output = re.sub(r'"answer":', '"output":', raw_output, flags=re.IGNORECASE)

#         instruction = raw_output[raw_output.lower().find('"instruction":') : raw_output.find('",')]
#         output = raw_output[raw_output.lower().find('"output":') : raw_output.find('."\n}')]
#         output+='."'

#         instruction = instruction.replace('"Instruction":', '"instruction":')
#         output = output.replace('"Output":', '"output":')

#         if "output" in instruction:
#             instruction = instruction[:instruction.find("output")].replace('."\n', '').replace('"\n  "', '"')

#         if '"instruction":' in output:
#             output = output[output.find('"instruction":') : output.find('",')]
#             output = output.replace('"instruction":', '')

#         #Remove final dot, to add later the ?.,
#         instruction = instruction.strip()[:-1] if instruction.strip().endswith(".") else instruction.strip()
#         instruction = instruction if instruction.endswith("?") else instruction+"?"
#         instruction = instruction+'.",'
#         #print(f"INS: {instruction}")

#         #return pre-defined null template if there "NULL on eather the insptructions or outputs"
#         instruction, output = self.check_for_null(instruction, output)

#         if "instruction" not in instruction:
#             instruction = '"instruction": "NULL?.",'
#             output = '"output": "NULL."'
#         elif "output" not in output:
#             output = '"output": "NULL".'
#             instruction = '"instruction": "NULL?.",'

#         if len(output.split("output")) > 2:
#             output = self.merge_outputs(output)

#         if "JSON object" in output:
#             output = '"output": "NULL."'

#         output = output.replace('"\n."', '."').replace('",\n', '')
#         output = output.replace('"\n', '"').replace('`', '').replace('{', '').replace('}', '').replace('"\n', '"').replace('".', '."').strip()

#         if ('{' in output):
#             output = output[:output.find('}')]
#         elif ('}' in output):
#             output = output[:output.find('{')]

#         output = self.remove_internal_double_virgolette(output, 'output')
#         instruction = self.remove_internal_double_virgolette(instruction, 'instruction')
#         return instruction, output, raw_output

#     def create_instruction_output_json(self, raw_outputs, json_output_name=None, verbose=False ):
#         chunks_to_repete=[]
#         n_max = self.n_max_chunks
#         all_outputs, str_json = [], []
#         for n in range(len(raw_outputs) if n_max == None else n_max):
#             instruction, output, _  = self.create_instruction_output(raw_outputs[n])
#             output_results=output.split('"output":')[-1]

#             if (len(output_results) < 15) | ("NULL" in output) | ("NULL" in instruction) | ("https//www.youtube" in output):
#                     instruction = '"instruction": "NULL.",'
#                     output = '"output": "NULL."'
#                     chunks_to_repete.append(n)

#             output = output.replace('"Question:', '')
#             instruction = instruction.replace('?"?."', '?."').replace('"?."', '').replace('"?.",', ',').replace('- "?."', ',')
#             if output.endswith('""'):
#                 output = output.replace('""', '"')

#             json_str = "{" + instruction + output + "}"
#             str_json.append(json_str)

#             try:
#                 parsed = json.loads(json_str.replace("\n", "").replace('"}``."}', '"}').replace('.""', '."'))        
#                 all_outputs.append(parsed)

#             except json.JSONDecodeError as e:
#                 if verbose:
#                     print("********************************")
#                     print(f"[!] JSON decode error at chunk {n}: {e}")
#                     print(f"Error input : {instruction}")
#                     print(f"Error output: {output}")
#                     print('json_str: ' ,json_str)
#                     print("********************************")
#                 else: 
#                     print(f"[!] JSON decode error at chunk {n}: {e}")

#         if json_output_name:
#             print(f"Saving outputs generated as: {json_output_name}")
#             with open(f"{json_output_name}.json", "w", encoding="utf-8") as f:
#                 json.dump(all_outputs, f, ensure_ascii=False, indent=2)

#         return all_outputs, str_json, chunks_to_repete
    
#     ################## CLEAN FINAL JSON QA FILE ##########################

#     # Cleaning/filter function
#     def is_valid(self, text):
#         if pd.isna(text):  
#             return False
#         if not isinstance(text, str):  # Check if not a string
#             return False

#         text = text.strip().lower()
#         # Check for known invalid values or if 'null' is present anywhere
#         invalid_keywords =["null", "none", "nan", "n/a"] 
#         if any(bad in text for bad in invalid_keywords):
#             return False

#         #minimum content length
#         if len(text) < 5:  
#             return False
#         return True

#     def filter_output(self, df):
#         # Apply filter to both columns
#         #print(df.head())
#         filtered_df = df[df['instruction'].apply(self.is_valid) & df['output'].apply(self.is_valid)].copy()
#         filtered_df = filtered_df.drop_duplicates()

#         cleaned_data = filtered_df.to_dict(orient="records")
#         return cleaned_data

#     def clean_json_file(self, all_outputs=None, json_file_name=None, 
#                         merge_with_existing_data=False, new_json_file_name=None):
#         if (all_outputs == None) & (json_file_name == None):
#             print(f"No input data was passed to the clean_json_file method. Add a list or a .json file to read.")
#         # elif (all_outputs != None) & (json_file_name != None) & (merge_with_existing_data == False):
#         #     print(f"Only 1 input argument is allowed when it's not desired to merge with current json {json_file_name}. Please select between reading a json file or using a current list with json format.")

#         if json_file_name != None:
#             json_file_name = json_file_name if json_file_name.endswith('.json') else json_file_name+'.json'
#             if os.path.exists(json_file_name):
#                 with open(json_file_name, "r", encoding="utf-8") as f:
#                     existing_data = json.load(f)
        
#         # if merge_with_existing_data:
#         #     if os.path.exists(json_file_name):
#         #         with open(json_file_name, "r", encoding="utf-8") as f:
#         #             existing_data = json.load(f)


#         elif all_outputs != None:
#             existing_data = all_outputs
#         else:
#             print("No input raw data was provided in clean_json_file method.")

#         df = pd.DataFrame(existing_data)
#         print('*********',df.head())
#         n_initial_samples = df.shape[0]

#         #returns a json format out of the input df after being preprocessed
#         cleaned_data = self.filter_output(df)
#         n_records = len(cleaned_data)

#         if (json_file_name!=None) & (new_json_file_name == None):
#             output_name = f"cleaned_{n_records}_{json_file_name}"
#         elif (json_file_name == None) & (new_json_file_name == None):
#             output_name= f"cleaned_{n_records}_from_raw_samples{n_initial_samples}"
#         elif (new_json_file_name!=None):
#             output_name = new_json_file_name

#         output_name = output_name if output_name.endswith('.json') else output_name+'.json'
            
#         with open(f"{output_name}", "w", encoding="utf-8") as f:
#             json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

#         return cleaned_data


# test=True
# if test:    
#     file_name = "llama_chunks_text_chuck_size250_overlap30.json"
#     # gc = GenerateQAContent(file_name)
#     # text = gc.get_text()
#     # print(len(text))
#     # raw = gc.generate_content(n_max_chunks=3)
#     #print(raw)
#     ##### Create a json from an pre-made raw generated QA
#     raw_content_name= "raw_outputs_samples1260_nreps3_new_tok256_chuck_size250_overlap30.json"
#     input_text_file_name=None
#     n_max_chunks=20

#     create_json = CreateJsonQA(raw_content_name=raw_content_name, input_text_file_name=input_text_file_name, n_max_chunks=n_max_chunks)

#     raw_outputs = create_json.get_raw_output()

#     json_output_name = "../data/test"
#     json_test, _, _ = create_json.create_instruction_output_json(raw_outputs, json_output_name=json_output_name,
#                                                            verbose=False)

#     #### Clean just created json

#     new_json_file_name = "../data/clean_test"
    
#     merge_with_existing_data = False
#     all_outputs = json_test
#     input_json_file_name = None#json_output_name

#     cleaned_json = create_json.clean_json_file(all_outputs=all_outputs, json_file_name=input_json_file_name, 
#                         merge_with_existing_data=merge_with_existing_data, new_json_file_name=new_json_file_name)

     