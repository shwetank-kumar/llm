from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

class TextGenerator():
    def __init__(self, checkpoint = "HuggingFaceH4/zephyr-7b-beta") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to("cuda")
        # You may want to use bfloat16 and/or move to GPU here
        
    def generate_text(self, messages):
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        print(self.tokenizer.decode(tokenized_chat[0]))
        generation_config = GenerationConfig(
            max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=self.model.config.eos_token_id,
            
        )
        outputs = self.model.generate(tokenized_chat, generation_config=generation_config) 
        print(self.tokenizer.decode(outputs[0]))

def main():
    text_generator = TextGenerator()
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate teacher",
        },
        {"role": "user", "content": "Write me an essay about impact of movies on violence in the society."},
    ]
    text_generator.generate_text(messages=messages)
    
    
if __name__ == "__main__":
    main()
