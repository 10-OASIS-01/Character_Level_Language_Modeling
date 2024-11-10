# scripts/train.py

import os
import sys
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random

# Append the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import create_datasets
from models.transformer import Transformer, ModelConfig
from models.rnn import RNN
from models.gru import GRU
from utils.helpers import evaluate
from utils.sampling import print_samples

def main():
    parser = argparse.ArgumentParser(description="Character-Level Language Modeling")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input text file.')
    parser.add_argument('--work_dir', type=str, default='work', help='Directory to save the model and logs.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--block_size', type=int, default=128, help='Context size for the model.')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of layers in the model.')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--n_embd', type=int, default=512, help='Embedding size.')
    parser.add_argument('--n_embd2', type=int, default=None, help='Secondary embedding size (if applicable).')
    parser.add_argument('--type', type=str, default='transformer', choices=['transformer', 'rnn', 'gru'],
                        help='Type of model to train.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint.')
    parser.add_argument('--sample_only', action='store_true', help='Only sample from the model without training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling parameter.')
    parser.add_argument('--sample_interval', type=int, default=5, help='Interval for generating samples during training.')
    args = parser.parse_args()
    print(vars(args))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    train_dataset, test_dataset = create_datasets(args.input_file, args.block_size)
    vocab_size = train_dataset.get_vocab_size()
    print(f"Dataset determined that: vocab_size={vocab_size}, block_size={args.block_size}")

    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_embd2=args.n_embd2
    )

    if args.type == 'transformer':
        model = Transformer(config)
    elif args.type == 'rnn':
        model = RNN(config)
    elif args.type == 'gru':
        model = GRU(config)
    else:
        raise ValueError(f'Model type {args.type} is not recognized')

    model.to(args.device)
    print(f"Model #params: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    start_epoch = 0
    if args.resume or args.sample_only:
        checkpoint_path = os.path.join(args.work_dir, f'{args.type}.pt')
        if os.path.exists(checkpoint_path):
            print("Resuming from existing model in the workdir")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_loss = checkpoint.get('best_loss', None)
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")

    if args.sample_only:
        print_samples(model, train_dataset, args, num=500)
        sys.exit()

    best_loss = None
    max_iterations = 2000

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # Randomly sample a subset of data for each epoch
        sampled_indices = random.sample(range(len(train_dataset)), min(len(train_dataset), max_iterations * args.batch_size))
        sampled_dataset = Subset(train_dataset, sampled_indices)
        train_loader = DataLoader(
            sampled_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        with tqdm(train_loader, unit="batch", total=max_iterations) as tepoch:
            for batch_idx, (X, Y) in enumerate(tepoch):
                if batch_idx >= max_iterations:
                    break
                X, Y = X.to(args.device), Y.to(args.device)
                logits, loss = model(X, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Loss/Train", loss.item(), global_step)

                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("Learning Rate", current_lr, global_step)
                tepoch.set_postfix(loss=loss.item())

        scheduler.step()
        train_loss = evaluate(model, train_dataset, args.device, batch_size=args.batch_size)
        test_loss = evaluate(model, test_dataset, args.device, batch_size=args.batch_size)
        writer.add_scalar("Loss/Eval_Train", train_loss, epoch)
        writer.add_scalar("Loss/Eval_Test", test_loss, epoch)
        writer.flush()
        print(f"Epoch {epoch + 1}/{args.num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        if best_loss is None or test_loss < best_loss:
            out_path = os.path.join(args.work_dir, "model.pt")
            print(f"Test loss {test_loss} is the best so far, saving model to {out_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': test_loss
            }, out_path)
            best_loss = test_loss

        if epoch % args.sample_interval == 0:
            print_samples(model, train_dataset, args, num=500)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

    writer.close()
    print("Training complete.")


if __name__ == '__main__':
    main()


"""
"transformer"
python scripts/train.py --input_file "data/input.txt" --work_dir "output_directory" --type "transformer" --block_size 64 --n_layer 4 --n_head 4 --n_embd 256 --batch_size 64 --num_epochs 10 --learning_rate 0.0001 --device "cuda" --seed 42 --sample_interval 2
"""


"""
"rnn"
python scripts/train.py --input_file "data/input.txt" --work_dir "output_directory_rnn" --type "rnn" --block_size 64 --n_layer 2 --n_embd 128 --n_embd2 128 --batch_size 64 --num_epochs 10 --learning_rate 0.001 --device "cuda" --seed 42 --sample_interval 2
"""

"""
"gru"
python scripts/train.py --input_file "data/input.txt" --work_dir "output_directory_gru" --type "gru" --block_size 64 --n_layer 3 --n_embd 256 --n_embd2 256 --batch_size 64 --num_epochs 10 --learning_rate 0.0005 --device "cuda" --seed 42 --sample_interval 2
"""

