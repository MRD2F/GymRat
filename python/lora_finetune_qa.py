
#GenAI imports 
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset
from huggingface_hub import login

#Data Preprocessing imports
import pandas as pd

from pathlib import Path
import sys
sys.path.append(str(Path('../python').resolve()))
from common_functions import get_private_keys

class FineTuneLora:
    def __init__(self, model_id, dataset_path, output_dir, lora_config, input_labels=["instruction", "output"]):
        self.model_id = model_id
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.lora_config = lora_config

        self.huggingface_hub_token = get_private_keys("huggingface_hub")
        login(token=self.huggingface_hub_token)

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self.get_model()
        self.input_labels = input_labels

        # Configure LoRA
        self.peft_model = get_peft_model(self.get_model(), self.lora_config)

    def get_model(self):
        quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  # also "fp4"
        bnb_4bit_compute_dtype="float16"
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True
        )
        return model

    def get_and_preprocess_dataset(self):
        # Load the dataset
        df = pd.read_csv(self.dataset_path)
        initial_count = len(df)
        df.dropna(subset=["answer", "focus_area"], inplace=True)
        final_count = len(df)

        print(f"Dropped {initial_count - final_count} rows with missing values.")
        print(f"Cleaned dataset now has {final_count} rows.")
        return df[["question", "answer"]]

    def convert_to_json(self, df, df_column_names):
        # Convert DataFrame to JSON format
        df.rename(columns={df_column_names[0]: "instruction", df_column_names[1]: "output"}, inplace=True)
        json_data = df.to_dict(orient='records')
        return json_data

    def get_json_lora_input(self,json_data):
        # Convert JSON data to a format suitable for LoRA training
        dataset = Dataset.from_list(json_data)   
        splits = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = splits["train"]
        test_dataset = splits["test"]    

        def format_example(example):
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
            return {"text": prompt}
        
        train_dataset = dataset.map(format_example)
        test_dataset = test_dataset.map(format_example)

        return train_dataset, test_dataset

    def train(self, df_column_names=["question", "answer"]  ):
        #json_dataset = self.convert_to_json(self.get_and_preprocess_dataset(), df_column_names)
        import json
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            json_dataset = json.load(f)
        train_dataset, test_dataset = self.get_json_lora_input(json_dataset)

        def tokenize_function(example):
            return self.tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
        

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./lora-llama-output",
            #per_device_train_batch_size=2 with gradient_accumulation_steps=4 is fine (effective batch size 8).
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            #learning_rate=2e-4, 2e-4 is a bit high for fine-tuning on a small dataset. It can cause unstable training or forgetting.
            learning_rate=1e-4,#1e-5,
            #num_train_epochs=3, #With only ~300 samples, 3 epochs might be too little.
            num_train_epochs=1,
            logging_dir="./logs",
            fp16=True,
            save_total_limit=2,
            logging_steps=10,
            save_steps=500,
            report_to="none", 
            #Add weight decay, Helps regularize, reduce overfitting.
            weight_decay=0.01,
            eval_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",

        )


        # Data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        # Train the model
        trainer.train()

    def save_model(self):
        self.peft_model.save_pretrained(self.output_dir)

test=True
if test:
    json_input_qa = "../data/clean_json_outputs_samples1711_nreps1_metallama_maxtoken128max_new_tokens512.json"

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # These depend on the model
        lora_dropout=0.05, # try slightly higher (0.1) if overfitting.
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    lora_fn = FineTuneLora(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_path=json_input_qa, 
            output_dir="../models/lora_finetuned_model",
            lora_config=lora_config)

    #df = lora_fn.get_and_preprocess_dataset()
    #print(df.head())
    #json = lora_fn.convert_to_json(df, ["question", "answer"])
    #json_formatted = lora_fn.get_json_lora_input(json)
    #print(lora_fn.convert_to_json(df, ["question", "answer"]))
    lora_fn.train(df_column_names=["question", "answer"])