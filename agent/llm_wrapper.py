# agent/llm_wrapper.py
import os
import re
import time
from typing import List, Tuple, Optional, Dict, Any

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, \
        StoppingCriteriaList

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class StopOnWords(StoppingCriteria):
    def __init__(self, tokenizer, stop_words: List[str], input_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_token_ids_list = []
        for word in stop_words:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if token_ids:
                self.stop_token_ids_list.append(torch.tensor(token_ids, dtype=torch.long))
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_ids = input_ids[0, self.input_len:]
        for stop_ids_tensor in self.stop_token_ids_list:
            stop_ids_tensor = stop_ids_tensor.to(generated_ids.device)
            if len(generated_ids) >= len(stop_ids_tensor):
                if torch.equal(generated_ids[-len(stop_ids_tensor):], stop_ids_tensor):
                    return True
        return False


class LlmWrapper:
    def __init__(self,
                 llm_provider: str = "local",
                 openai_model_name: Optional[str] = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 local_model_id: Optional[str] = "microsoft/Phi-3-mini-4k-instruct",
                 local_model_device_map: Optional[str] = "auto",
                 local_model_trust_remote_code: bool = True,
                 local_model_torch_dtype: Optional[str] = "auto",
                 local_model_quantization: Optional[str] = None,
                 temperature: float = 0.1,
                 max_new_tokens: int = 512,
                 max_tokens_critique: int = 400):

        self.llm_provider = llm_provider.lower()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_tokens_critique = max_tokens_critique

        self.openai_client = None
        self.local_model = None
        self.local_tokenizer = None

        if self.llm_provider == "openai":
            if not OPENAI_AVAILABLE: raise ImportError("OpenAI lib not installed for 'openai' provider.")
            self.openai_model_name = openai_model_name
            used_api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not used_api_key:
                print("WARNING: OpenAI API key not provided.")
            else:
                try:
                    self.openai_client = openai.OpenAI(api_key=used_api_key)
                    print(f"OpenAI client initialized for model: {self.openai_model_name}")
                except Exception as e:
                    print(f"ERROR: Failed to initialize OpenAI client: {e}")

        elif self.llm_provider == "local":
            if not TRANSFORMERS_AVAILABLE: raise ImportError("Transformers lib not installed for 'local' provider.")
            if not local_model_id: raise ValueError("local_model_id required for 'local' provider.")

            print(f"Initializing local Hugging Face model: {local_model_id}")
            try:
                quant_config = None
                if local_model_quantization:
                    if local_model_quantization == "4bit":
                        quant_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                        print("Using 4-bit quantization.")
                    elif local_model_quantization == "8bit":
                        quant_config = BitsAndBytesConfig(load_in_8bit=True)
                        print("Using 8-bit quantization.")

                dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
                             "auto": "auto"}
                torch_dtype_val = dtype_map.get(str(local_model_torch_dtype).lower(), "auto")

                self.local_tokenizer = AutoTokenizer.from_pretrained(local_model_id,
                                                                     trust_remote_code=local_model_trust_remote_code)
                if self.local_tokenizer.pad_token_id is None:
                    self.local_tokenizer.pad_token_id = self.local_tokenizer.eos_token_id
                    print(f"Tokenizer pad_token_id set to eos_token_id ({self.local_tokenizer.eos_token_id}).")

                model_kwargs = {
                    "device_map": local_model_device_map if torch.cuda.is_available() else "cpu",
                    "trust_remote_code": local_model_trust_remote_code,
                    "quantization_config": quant_config,
                    "attn_implementation": "eager"
                }
                if isinstance(torch_dtype_val, torch.dtype):
                    model_kwargs["torch_dtype"] = torch_dtype_val

                self.local_model = AutoModelForCausalLM.from_pretrained(
                    local_model_id,
                    **model_kwargs
                )
                self.local_model.eval()
                print(
                    f"Local model '{local_model_id}' loaded. Main device: {next(self.local_model.parameters()).device}.")
            except Exception as e:
                print(f"ERROR loading local model '{local_model_id}': {e}")
                import traceback;
                traceback.print_exc()
                raise RuntimeError(f"Failed to load local model: {e}") from e
        else:
            raise ValueError(f"Unsupported llm_provider: {self.llm_provider}.")

    def _apply_chat_template_if_needed(self, prompt: str) -> str:
        # This function helps format the prompt correctly for certain models, like Phi-3.
        # It assumes the tokenizer has a chat template.
        if hasattr(self.local_tokenizer, 'apply_chat_template') and \
                ("phi-3" in self.local_tokenizer.name_or_path.lower() or \
                 "llama-3" in self.local_tokenizer.name_or_path.lower() or \
                 "gemma" in self.local_tokenizer.name_or_path.lower()):  # Add other models if they benefit
            messages = [{"role": "user", "content": prompt}]
            try:
                # add_generation_prompt=True is important for instruct models to know it should generate
                return self.local_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                print(
                    f"Warning: Could not apply chat template for {self.local_tokenizer.name_or_path}: {e}. Using raw prompt.")
        return prompt

    def _call_openai_api(self, prompt_content: str, max_tokens_val: int, temperature_val: float,
                         stop_seq: Optional[List[str]], num_cand: int) -> List[str]:
        if not self.openai_client: return [f"Error: OpenAI client not init."] * num_cand
        messages = [{"role": "user", "content": prompt_content}]
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model_name, messages=messages, max_tokens=max_tokens_val,
                temperature=temperature_val, n=num_cand, stop=stop_seq)
            return [choice.message.content.strip() for choice in response.choices]
        except Exception as e:
            raise RuntimeError(f"OpenAI API Error: {e}") from e

    def _call_local_model_api(self, prompt_content: str, max_new_tokens_val: int, temperature_val: float,
                              stop_seq: Optional[List[str]], num_cand: int) -> List[str]:
        if not self.local_model or not self.local_tokenizer:
            return [f"Error: Local model/tokenizer not init."] * num_cand

        templated_prompt = self._apply_chat_template_if_needed(prompt_content)

        model_max_len = getattr(self.local_model.config, 'max_position_embeddings', 4096)
        # Ensure tokenizer_max_len is positive
        tokenizer_max_len = max(1, model_max_len - max_new_tokens_val - 20)  # Buffer of 20 tokens

        inputs = self.local_tokenizer(templated_prompt, return_tensors="pt", truncation=True,
                                      max_length=tokenizer_max_len)
        inputs = inputs.to(self.local_model.device)
        input_len = inputs.input_ids.shape[1]

        stop_criteria_list = None
        if stop_seq:
            stop_criteria_list = StoppingCriteriaList([StopOnWords(self.local_tokenizer, stop_seq, input_len)])

        do_sample = temperature_val > 0.01
        gen_kwargs = {
            "max_new_tokens": max_new_tokens_val,
            "pad_token_id": self.local_tokenizer.pad_token_id,
            "eos_token_id": self.local_tokenizer.eos_token_id,
            "do_sample": do_sample,
            "use_cache": True,  # Explicitly set, though it's default
        }
        if stop_criteria_list:
            gen_kwargs["stopping_criteria"] = stop_criteria_list

        if do_sample:
            gen_kwargs["temperature"] = temperature_val
            gen_kwargs["top_p"] = 0.9  # A common default for sampling

        responses = []
        for _ in range(num_cand):
            with torch.no_grad():
                outputs = self.local_model.generate(**inputs, **gen_kwargs)
            response_ids = outputs[0][input_len:]
            text = self.local_tokenizer.decode(response_ids, skip_special_tokens=True)
            if stop_seq:
                for s_word in stop_seq:
                    if s_word in text:
                        text = text.split(s_word)[0]
                        break
            responses.append(text.strip())
        return responses

    def _parse_thought_action(self, raw_output: str) -> Tuple[str, str, str]:
        thought, action_type, action_input = "", "", ""
        thought_match = re.search(r"Thought:(.*)", raw_output, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought_text_full = thought_match.group(1).strip()
            action_match = re.search(r"Action:\s*([^\n]+)", thought_text_full, re.IGNORECASE)
            if action_match:
                thought = thought_text_full[:action_match.start()].strip()
                action_type = action_match.group(1).strip()
                action_input_text_full = thought_text_full[action_match.end():].strip()
                action_input_match = re.search(r"Action Input:\s*(.*)", action_input_text_full,
                                               re.DOTALL | re.IGNORECASE)
                action_input = action_input_match.group(
                    1).strip() if action_input_match else action_input_text_full.strip()
            else:
                thought = thought_text_full
        else:
            action_match_direct = re.search(r"Action:\s*([^\n]+)", raw_output, re.IGNORECASE)
            if action_match_direct:
                action_type = action_match_direct.group(1).strip()
                action_input_match_direct = re.search(r"Action Input:\s*(.*)", raw_output[action_match_direct.end():],
                                                      re.DOTALL | re.IGNORECASE)
                action_input = action_input_match_direct.group(1).strip() if action_input_match_direct else ""
            else:
                thought = raw_output.strip()
                action_type = "ParsingError"

        if thought and not action_type:
            action_type = "ContinueThought"

        if action_type and '\n' in action_type:
            action_type = action_type.split('\n')[0].strip()

        return thought, action_type, action_input

    def generate_thought_action(self, prompt: str) -> Tuple[str, str, str]:
        try:
            if self.llm_provider == "openai":
                responses = self._call_openai_api(prompt, self.max_new_tokens, self.temperature,
                                                  ["Observation:", "\nThought:"], 1)
            elif self.llm_provider == "local":
                responses = self._call_local_model_api(prompt, self.max_new_tokens, self.temperature,
                                                       ["Observation:", "\nThought:"], 1)
            else:
                raise RuntimeError(f"Invalid provider: {self.llm_provider}")
            if not responses or not responses[0].strip(): return "Error: Empty LLM response.", "Error", ""
            return self._parse_thought_action(responses[0])
        except Exception as e:
            print(f"Exception in generate_thought_action: {e}")
            import traceback;
            traceback.print_exc()  # Ensure full traceback is printed
            return f"Error in LLM call: {e}", "Error", ""

    def generate_candidates(self, prompt: str, num_cand: int) -> List[str]:
        if num_cand <= 0: return []
        temp = min(self.temperature + 0.2, 0.9) if self.temperature < 0.7 else self.temperature
        try:
            if self.llm_provider == "openai":
                res = self._call_openai_api(prompt, self.max_new_tokens, temp, ["Observation:", "\nThought:"], num_cand)
            elif self.llm_provider == "local":
                res = self._call_local_model_api(prompt, self.max_new_tokens, temp, ["Observation:", "\nThought:"],
                                                 num_cand)
            else:
                raise RuntimeError(f"Invalid provider: {self.llm_provider}")
            return list(dict.fromkeys(c for c in res if c.strip()))
        except Exception as e:
            print(f"ERROR in generate_candidates: {e}")
            return []

    def generate_critique(self, prompt: str) -> str:
        try:
            if self.llm_provider == "openai":
                res = self._call_openai_api(prompt, self.max_tokens_critique, self.temperature, None, 1)
            elif self.llm_provider == "local":
                res = self._call_local_model_api(prompt, self.max_tokens_critique, self.temperature, None, 1)
            else:
                raise RuntimeError(f"Invalid provider: {self.llm_provider}")
            if not res or not res[0].strip(): return "Error: Empty critique from LLM."
            return res[0]
        except Exception as e:
            return f"Error generating critique: {e}"
