import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class Benchmark:

    def __init__(self, hf_model):
        # self.strategy = strategy
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(hf_model, device_map="auto", torch_dtype=torch.bfloat16)

    def analyze(self,text_data):
        tokens = self.tokenizer(text_data, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**tokens)

        embedding = outputs[0].to("cpu")
        self_similarity_matrix = cosine_similarity(embedding.squeeze(), embedding.squeeze())

        return self_similarity_matrix

def plot_self_similarity(matrix): 
    # Assuming you already have the self_similarity_matrix from the previous code

    # Plot the self-similarity matrix as a heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap='Blues', interpolation='nearest')

    # Add colorbar
    plt.colorbar()

    # Display the plot
    plt.title('2D Self-Similarity Matrix')
    plt.show()