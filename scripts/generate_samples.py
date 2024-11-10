# scripts/generate_samples.py

import os
import sys
import argparse
import torch

# Append the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer import Transformer, ModelConfig
from utils.sampling import print_samples
from data.dataset import create_datasets


def main():
    parser = argparse.ArgumentParser(description="Generate Samples from Trained Model")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input text file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on.')
    parser.add_argument('--num_chars', type=int, default=500, help='Number of characters to generate.')
    parser.add_argument('--block_size', type=int, default=64, help='Context size for the model.')  # Add block_size argument
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling parameter.')
    parser.add_argument('--start_string', type=str, default="", help='String to prime the generation.')

    args = parser.parse_args()

    # Load dataset to get vocab and mappings using the block size from args
    train_dataset, _ = create_datasets(args.input_file, block_size=args.block_size)

    # Load model configuration and weights
    config = ModelConfig(
        vocab_size=train_dataset.get_vocab_size(),
        block_size=args.block_size,  # Use block_size from args
        n_layer=4,
        n_head=4,
        n_embd=256  # Must match the embedding size used during training
    )
    model = Transformer(config)

    # Load checkpoint and extract model weights
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])  # Load only model weights
    model.to(args.device)

    # Generate and print samples with the start string
    print_samples(model, train_dataset, args, start_string=args.start_string, num=args.num_chars)


if __name__ == '__main__':
    main()


"""
python scripts/generate_samples.py \
    --input_file "data/input.txt" \
    --model_path 'output_directory/model.pt' \
    --device 'cuda' \
    --num_chars 500 \
    --block_size 64 \
    --top_k 40 \
    --start_string "ROMEO: "
"""