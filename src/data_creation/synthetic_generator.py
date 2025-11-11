# from config.env_loader import setup_env
# setup_env()

from tqdm import tqdm
import mlflow
from datetime import datetime
import yaml
from dataclasses import dataclass
from typing import List
#works when in terminal root project: 
# export PYTHONPATH=$PYTHONPATH:/home/mrosaria/Projects/NLP/GymRat/src
from utils import get_private_key
from huggingface_hub import login
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
import time
import json
from json_repair import repair_json


@dataclass
class QAGenerationConfig:
    model_id: str
    experiment_name: str
    use_bnb: bool
    batch_size: int
    n_reps: int
    temperature: float
    top_p: float
    max_tokens: int
    sample_i: int
    sample_f: int

    @staticmethod
    def from_yaml(path: str) -> "QAGenerationConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        gen = data["generation"]
        return QAGenerationConfig(
            model_id=data["model_id"],
            use_bnb=data["use_bnb"],
            experiment_name=data["experiment_name"],
            batch_size=gen["batch_size"],
            n_reps=gen["n_reps"],
            temperature=gen["temperature"],
            top_p=gen["top_p"],
            max_tokens=gen["max_tokens"],
            sample_i=gen["sample_i"],
            sample_f=gen["sample_f"],
        )

class SyntheticQAGenerator:
    def __init__(self, chuncked_data_file_name: str, config: QAGenerationConfig):
        self.cfg = config    
        self.chuncked_data_file_name = chuncked_data_file_name
        self.output_file_name = f"synthetic_QA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        login(token=get_private_key("HF_TOKEN"))
        

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)

        # Ensure EOS token is set (usually already set)
        tokenizer.eos_token = tokenizer.eos_token or "</s>"
        # Set pad token to EOS if not defined
        #Using pad_token = eos_token is a common workaround for models like LLaMA.
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        #left using commonly for generation and training, right for inference
        tokenizer.padding_size="right"
        return tokenizer


    def _load_model(self):
        kwargs = {"device_map": "auto"}

        if self.cfg.use_bnb:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
            )

        return AutoModelForCausalLM.from_pretrained(self.cfg.model_id, **kwargs)

    def create_pipeline(self):
        tokenizer = self._load_tokenizer()
        pipe = pipeline("text-generation",
                         model=self._load_model(), 
                         tokenizer=tokenizer,
                         dtype="auto", device_map="auto")
        return pipe, tokenizer
    

    def _get_text(self) -> List[str]:
        with open(self.chuncked_data_file_name , 'r') as f:
            llama_chunks = f.readlines()
        return llama_chunks 
        
    def _get_samples(self, n_chunks_intervals : List[int] = None) -> List[str]:
        text_chunks = self._get_text()
        samples = text_chunks[:] if n_chunks_intervals == None else text_chunks[n_chunks_intervals[0]:n_chunks_intervals[1]]
        return samples

    @staticmethod
    def _get_prompt_fn(text : str) -> List[dict]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise and helpful medical tutor. "
                    "Based on the provided text, generate a JSON object with exactly ONE question (as 'instruction') and ONE answer (as 'output').\n\n"
                    "- The content must relate to health, exercise, sports, fitness, or physiotherapy.\n"
                    "- Do not include multiple questions or answers.\n"
                    "- Do not repeat the instruction in the output.\n"
                    "- The output must contain a thorough and detailed, multi-paragraph question (as 'instruction') and answer (as 'output').\n"
                    "- If the text is not relevant, return: {\"instruction\": \"NULL\", \"output\": \"NULL\"}\n\n"
                    "- Respond ONLY with the JSON object. Do NOT include any explanation or commentary."
                ), #multi-paragraph answer
            },
            {
                "role": "user",
                "content": text.strip()
            },
        ]
        return messages
    
    def _get_prompt_with_token_chat_template_fn(self, text : str, tokenizer: AutoTokenizer) -> List[str]:
        # Use the tokenizer’s built-in chat template rendering
        prompt = self._get_prompt_fn(text)
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    
    @staticmethod
    def _chunk_list(lst: list, size: int) -> List[list]:
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), size):
            yield lst[i:i+size]

    #Prefix helper methods with _ if they’re meant for internal use.
    def _log_run_params(self):
        cfg = self.cfg
        for key, value in cfg.__dict__.items():
            mlflow.log_param(key, value)


    def generate(self):
        chat, tokenizer = self.create_pipeline()
        cfg = self.cfg
        samples = self._get_samples()

        json_output_name = f"../../data/synthetic/synthetic_QA_{datetime.now():%Y%m%d_%H%M%S}_samples{cfg.sample_i}-{cfg.sample_f}.json"

        results = []

        mlflow.set_experiment(cfg.experiment_name)
        with mlflow.start_run(run_name=f"qa_batch_{cfg.sample_i}_{cfg.sample_f}"):
            self._log_run_params()

            #total_prompt_tokens, total_completion_tokens = [], []

            # ==== MAIN GENERATION LOOP ====
            for batch in tqdm(self._chunk_list(samples[cfg.sample_i : cfg.sample_f], cfg.batch_size)):
                batch_prompts = [self._get_prompt_with_token_chat_template_fn(sample, tokenizer) for sample in batch]
                # Iterate over each sample in the batch
                for prompt in batch_prompts:
                    for _ in range(cfg.n_reps):
                        try:
                            chat_completion = chat(prompt,
                                                   max_new_tokens=cfg.max_tokens,
                                                   #max_length=cfg.max_tokens,
                                                   do_sample=True,
                                                   temperature=cfg.temperature,
                                                   top_p=cfg.top_p,
                                                   num_return_sequences=cfg.n_reps, 
                                                   truncation=True)

                            # Parse output
                            raw_output = chat_completion[0]["generated_text"]

                            #Metadata included in the Databricks / OpenAI-style APIs
                            #usage = getattr(raw_output, "usage", None)

                            if "<|start_header_id|>assistant<|end_header_id|>" in raw_output:
                                raw_output = raw_output.split("<|start_header_id|>assistant<|end_header_id|>")[-1]

                            # Optional cleanup
                            content = raw_output.strip().replace("<|eot_id|>", "")

                            # Parse if it’s valid JSON
                            instruction, output = None, content
                            try:
                                repaired = repair_json(content)
                                parsed = json.loads(repaired)
                                instruction = parsed.get("instruction", "")
                                output = parsed.get("output", "")
                                      
                            except json.JSONDecodeError:
                                # instruction, output = None, content
                                print("⚠️ Output was not valid JSON, Could not parse even after repair:", content)

                        
                            results.append({
                                "prompt": prompt,
                                "instruction": instruction,
                                "output": output,
                                # "usage": {
                                #     "prompt_tokens": getattr(usage, "prompt_tokens", None),
                                #     "completion_tokens": getattr(usage, "completion_tokens", None),
                                #     "total_tokens": getattr(usage, "total_tokens", None)
                                # }
                            })

                            # Collect token metrics
                            # if usage:
                            #     total_prompt_tokens.append(usage.prompt_tokens)
                            #     total_completion_tokens.append(usage.completion_tokens)

                            time.sleep(0.5)  # avoid hitting rate limits
                        except Exception as e:
                            print(f"Error generating completion for prompt: {prompt}")
                            print(e)

            # ==== SAVE OUTPUT LOCALLY ====
            with open(json_output_name, "w") as f:
                json.dump(results, f, indent=2)


            # ==== LOG METRICS & ARTIFACTS ====
            # if total_prompt_tokens:
            #     mlflow.log_metric("avg_prompt_tokens", np.mean(total_prompt_tokens))
            #     mlflow.log_metric("avg_completion_tokens", np.mean(total_completion_tokens))
            #     mlflow.log_metric("total_generations", len(results))

            # Upload JSON output as artifact
            mlflow.log_artifact(json_output_name)

            mlflow.set_tag("dataset", "synthetic_QA_physio")
            mlflow.set_tag("run_type", "generation")
            mlflow.set_tag("status", "completed")

        print(f"✅ Generation completed. Logged results in MLflow experiment.")        

       

if __name__ == "__main__":
    file_name = "../../data/processed/rebuilding_milo_chunks_docling_max_tokens128_min_tokens50_meta_llama3p18B.txt" 
    config = QAGenerationConfig.from_yaml("../../config/qa_generation.yaml")
    generator = SyntheticQAGenerator(file_name, config=config)

    generator.generate()

