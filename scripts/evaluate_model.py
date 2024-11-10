# scripts/evaluate_model.py

import os
import sys
import math
import torch
import nltk
import json
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
# Append the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer import Transformer, ModelConfig
from models.rnn import RNN
from models.gru import GRU
from utils.sampling import generate
from data.dataset import create_datasets

nltk.download('punkt')
nltk.download("punkt_tab")


def calculate_ppl(model, dataset, device, batch_size=64):
    """
    Calculate perplexity (PPL) on a dataset.

    Args:
        model (torch.nn.Module): Trained language model.
        dataset (torch.utils.data.Dataset): Dataset for evaluation.
        device (str): Device for computation (e.g., "cuda" or "cpu").
        batch_size (int): Batch size for data loading.

    Returns:
        float: The perplexity of the model on the dataset.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            total_loss += loss.item() * X.size(0)
            total_tokens += X.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def evaluate_bleu_rouge(model, dataset, device, start_string, target_text, num_tokens=500, temperature=1.0, top_k=None):
    """
    Generate text and calculate BLEU and ROUGE scores against the target text.

    Args:
        model: The language model.
        dataset: The dataset object for vocabulary mappings.
        device: Device for model computation.
        start_string: The starting text for generation.
        target_text: The reference text to compare against.
        num_tokens: Number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.

    Returns:
        dict: A dictionary with BLEU and ROUGE scores.
    """
    idx = torch.tensor([dataset.stoi[s] for s in start_string], dtype=torch.long).to(device)

    generated_indices = generate(
        model=model,
        idx=idx,
        max_new_tokens=num_tokens,
        device=device,
        temperature=temperature,
        top_k=top_k
    )
    generated_text = ''.join([dataset.itos[i] for i in generated_indices.tolist()])

    # Tokenize the generated and target text
    generated_tokens = nltk.word_tokenize(generated_text)
    target_tokens = nltk.word_tokenize(target_text)

    # Calculate BLEU score
    bleu_score = sentence_bleu([target_tokens], generated_tokens, weights=(0.5, 0.5), smoothing_function=SmoothingFunction().method1)

    # Calculate ROUGE score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(' '.join(target_tokens), ' '.join(generated_tokens))

    scores = {
        'BLEU': bleu_score,
        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        'ROUGE-L': rouge_scores['rougeL'].fmeasure
    }

    return scores, generated_text


import os
import json
import torch

def main(model_path, input_file, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Main function to evaluate the trained model.

    Args:
        model_path: Path to the saved model checkpoint.
        input_file: Path to the dataset file.
        device: Device to use for computation.
    """
    # Load dataset
    train_dataset, test_dataset = create_datasets(input_file, block_size=64)
    vocab_size = train_dataset.get_vocab_size()
    print(f"Loaded dataset with vocab_size={vocab_size}")

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Use config from checkpoint if available, otherwise create a default one
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = ModelConfig(
            vocab_size=vocab_size,
            block_size=64,
            n_layer=4,
            n_head=4,
            n_embd=256
        )

    # Initialize model based on config
    model_type = getattr(config, 'model_type', 'transformer')
    if model_type == 'transformer':
        model = Transformer(config)
    elif model_type == 'rnn':
        model = RNN(config)
    elif model_type == 'gru':
        model = GRU(config)
    else:
        raise ValueError("Invalid model type in checkpoint.")

    # Load model state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("Model loaded successfully.")

    # Load evaluation data
    eval_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'evaluate_string.json')
    with open(eval_file, 'r') as f:
        eval_data = json.load(f)

    total_bleu = 0
    total_rouge_1 = 0
    total_rouge_2 = 0
    total_rouge_L = 0
    total_ppl = 0  # Initialize total perplexity
    num_examples = len(eval_data)

    for example in eval_data:
        start_string = example['start_string']
        target_text = example['target_text']
        target_tokens = target_text.split()
        num_tokens = len(target_tokens)

        scores, generated_text = evaluate_bleu_rouge(
            model=model,
            dataset=train_dataset,
            device=device,
            start_string=start_string,
            target_text=target_text,
            num_tokens=num_tokens,
            temperature=1.0,
            top_k=None
        )

        # Calculate Perplexity for this example
        example_ppl = calculate_ppl(model, test_dataset, device)
        total_ppl += example_ppl

        print(f"\nGenerated Text for Example:")
        print(generated_text)
        print("\nScores for Example:")
        print(f"BLEU Score: {scores['BLEU']:.4f}")
        print(f"ROUGE-1 F1: {scores['ROUGE-1']:.4f}")
        print(f"ROUGE-2 F1: {scores['ROUGE-2']:.4f}")
        print(f"ROUGE-L F1: {scores['ROUGE-L']:.4f}")
        print(f"Perplexity (PPL) for Example: {example_ppl:.4f}")

        total_bleu += scores['BLEU']
        total_rouge_1 += scores['ROUGE-1']
        total_rouge_2 += scores['ROUGE-2']
        total_rouge_L += scores['ROUGE-L']

    # Calculate average scores
    avg_bleu = total_bleu / num_examples
    avg_rouge_1 = total_rouge_1 / num_examples
    avg_rouge_2 = total_rouge_2 / num_examples
    avg_rouge_L = total_rouge_L / num_examples
    avg_ppl = total_ppl / num_examples

    print("\nAverage Evaluation Scores:")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average ROUGE-1 F1: {avg_rouge_1:.4f}")
    print(f"Average ROUGE-2 F1: {avg_rouge_2:.4f}")
    print(f"Average ROUGE-L F1: {avg_rouge_L:.4f}")
    print(f"Average Perplexity (PPL): {avg_ppl:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Trained Language Model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model checkpoint.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input text file for dataset creation.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the evaluation on.')
    args = parser.parse_args()

    main(args.model_path, args.input_file)

"""
python scripts/evaluate_model.py --model_path output_directory/model.pt --input_file data/input.txt 
"""