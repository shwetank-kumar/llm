from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():

    checkpoint = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")  # You may want to use bfloat16 and/or move to GPU here

    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate teacher",
        },
        {"role": "user", "content": "Write me an essay about impact of movies on violence in the society."},
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    print(tokenizer.decode(tokenized_chat[0]))

    outputs = model.generate(tokenized_chat, max_new_tokens=500) 
    print(tokenizer.decode(outputs[0]))

    
if __name__ == "__main__":
    main()
