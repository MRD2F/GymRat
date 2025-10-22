from pathlib import Path
import sys
sys.path.append(str(Path('../python').resolve()))
from create_clean_chunks import *
from transformers import pipeline, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
from transformers import AutoTokenizer

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print('nevermid')

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
length_function=llama_token_len
cc = CreateChunks(file, chunk_size,chunk_overlap, length_function )
print('nevermid')
llama_chunks=cc.get_clean_chunks()
print(f"Total chunks created: {len(llama_chunks)}")

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",  # You can also use "fp4"
#     bnb_4bit_compute_dtype="float16"
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",
#     quantization_config=quant_config,
#     trust_remote_code=True
# )

# qa_gen = pipeline("text-generation", model=model, tokenizer=get_tokenizer(),  max_new_tokens=256)

# def get_prompt_v2(text):
#     messages = [
#     {
#         "role": "system",
#         "content": (
#             "You are a concise and helpful medical tutor. "
#             "Based on the provided text, generate a JSON object with exactly ONE question (as 'instruction') and ONE answer (as 'output').\n\n"
#             "- The content must relate to health, exercise, sports, fitness, or physiotherapy.\n"
#             "- Do not include multiple questions or answers.\n"
#             "- Do not repeat the instruction in the output.\n"
#             "- Keep the output brief and informative.\n"
#             "- If the text is not relevant, return: {\"instruction\": \"NULL\", \"output\": \"NULL\"}\n\n"
#             "- Respond ONLY with the JSON object. Do NOT include any explanation or commentary."
#         ),
#     },
#     {
#         "role": "user",
#         "content": text.strip()
#     },
# ]
#     tokenizer = get_tokenizer()
#     return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



# batch_size = 16
# from tqdm import tqdm

# text_chunks = llama_chunks
# raw_outputs = []
# n_repetitions = 3
# samples = text_chunks[100:105]

# for i in tqdm(range(0, len(samples), batch_size)):
#     print(f"i: {i}")
#     batch = samples[i:i + batch_size]
#     print(f"Processing batch with samples {i:i + batch_size} ")

#     for _ in range(n_repetitions):  # Repeat generation 3 times per batch
#         prompt = [get_prompt_v2(chunk) for chunk in batch]
        
#         raw_output = qa_gen(
#             prompt, 
#             max_new_tokens=256, 
#             do_sample=True,
#             temperature=0.7,
#             top_k=50,
#             top_p=0.95
#         )
        
#         raw_outputs.extend([o[0]["generated_text"] for o in raw_output])


# import json

# # Save to a file
# with open(f"raw_outputs_samples{len(samples)}_nreps{n_repetitions}_new_tok{256}_chuck_size{250}_overlap{30}.json", "w") as f:
#     json.dump(raw_outputs, f)
