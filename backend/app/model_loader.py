import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelService:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("MODEL_PATH", "./model_weights")
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        """Loads the model and tokenizer into memory once on startup."""
        if not os.path.exists(self.model_path):
            print(f"[WARNING] Model path '{self.model_path}' not found.")
            print("[INFO] You can set the MODEL_PATH environment variable or place files in ./model_weights.")
            return False

        print(f"[INFO] Loading model from '{self.model_path}' onto device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model with float16 if on GPU to save memory, else float32 on CPU
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        attn_impl = "sdpa" if self.device == "cuda" else "eager"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch_dtype,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        self.model.eval()

        # Warmup GPU kernels so first user request doesn't suffer JIT compile lag
        if self.device == "cuda":
            print("[INFO] Warming up CUDA kernels...")
            try:
                dummy_input = torch.tensor([[self.tokenizer.bos_token_id or 0]], device=self.device)
                with torch.inference_mode():
                    self.model.generate(dummy_input, max_new_tokens=2, pad_token_id=self.tokenizer.eos_token_id)
            except Exception as e:
                print(f"[WARNING] Warmup skipped: {e}")

        print("[INFO] Model loaded successfully.")
        return True

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2
    ):
        """Runs inference given an optional prompt and sampling parameters."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Please ensure model files exist in the specified path.")

        start_time = time.perf_counter()
        prompt_text = prompt if prompt is not None else ""

        # Tokenize input or initialize with BOS/start token if empty
        if prompt_text.strip():
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            input_token_count = inputs["input_ids"].shape[1]
        else:
            encoded = self.tokenizer(prompt_text, return_tensors="pt")
            if encoded["input_ids"].shape[1] > 0:
                inputs = {k: v.to(self.device) for k, v in encoded.items()}
                input_token_count = inputs["input_ids"].shape[1]
            else:
                bos_token_id = (
                    self.tokenizer.bos_token_id
                    if self.tokenizer.bos_token_id is not None
                    else (self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0)
                )
                input_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=self.device)
                inputs = {"input_ids": input_ids}
                if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
                    inputs["attention_mask"] = torch.ones_like(input_ids)
                input_token_count = 1

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        # Generate tokens
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0.0,
                pad_token_id=pad_token_id
            )

        latency = time.perf_counter() - start_time
        total_tokens = outputs.shape[1]
        new_tokens_count = max(0, total_tokens - input_token_count)

        # Decode continuation
        generated_ids = outputs[0][input_token_count:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "prompt": prompt_text,
            "generated_text": generated_text,
            "tokens_generated": new_tokens_count,
            "latency_seconds": round(latency, 4)
        }

# Singleton instance
model_service = ModelService()
