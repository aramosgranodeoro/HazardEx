import asyncio
import gc
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel

class VLMModelManager:
    def __init__(self):
        self.base_model_path = "unsloth/llava-1.5-7b-hf-bnb-4bit"
        self.processor = None
        self.model = None
        self.lock = asyncio.Lock()

    def _load(self):
        if self.model is not None:
            return  # ya cargado
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.base_model_path)

        base_model = LlavaForConditionalGeneration.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        model = PeftModel.from_pretrained(
            base_model,
            "C:/dev/fine_tuning/violencia/llava-violence-adapter_v2",
            adapter_name="violence",
        )
        model.load_adapter(
            "C:/dev/fine_tuning/coches/llava-crash-adapter_v1",
            adapter_name="traffic_accident",
        )
        self.model = model

    def _unload(self):
        if self.model is None:
            return
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    async def generate(self, category: str, image, prompt: str) -> str:
        async with self.lock:
            self._load()
            self.model.set_adapter(category)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            formatted_prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt").to("cuda")
            output = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
            result = self.processor.decode(output[0], skip_special_tokens=True)
            return result

    async def release(self):
        async with self.lock:
            self._unload()

vlm_manager = VLMModelManager()