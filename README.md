# Character-Level Language Modeling Project

This project implements character-level language modeling using Transformer, RNN, and GRU architectures, specifically trained on the tiny Shakespeare dataset. The models learn to predict the next character in a sequence, enabling the generation of text that stylistically resembles Shakespeare's writing.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Generating Samples](#generating-samples)
  - [Evaluation](#evaluation)
- [TensorBoard Visualization](#tensorboard-visualization)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates character-level language modeling, generating text one character at a time based on historical context. Three neural network architectures are used:

- **Transformer**: Utilizes self-attention for efficient context modeling across long character sequences.
- **RNN**: A traditional recurrent neural network that learns character dependencies in sequence.
- **GRU**: A variation of RNN that uses gates to manage long-term dependencies more effectively.

The models are trained on the Tiny Shakespeare dataset, a compact collection of Shakespeare's works, to produce text samples that mimic the style and language of Shakespeare.

## Features

- Modular codebase with separate components for data loading, model architectures, utilities, and training/generation scripts.
- Supports multiple architectures for comparison (Transformer, RNN, GRU).
- Configurable hyperparameters using command-line arguments for flexible experimentation.
- Integration with TensorBoard for real-time visualization of training metrics.
- Scripts for easy model training, sample generation, and evaluation.
- Unit tests to ensure model and code reliability.

## Project Structure

```
your_project/
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate_string.json
│   └── input.txt
├── models/
│   ├── __init__.py
│   ├── transformer.py
│   ├── rnn.py
│   └── gru.py
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   └── sampling.py
├── scripts/
│   ├── __init__.py
│   ├── train.py
│   ├── generate_samples.py
│   └── evaluate_model.py
├── tests/
│   └── test_models.py
├── output_directory/
│   ├── events.out.tfevents.1731235237.LAPTOP-H8HJ1JCA.35892.0
│   ├── events.out.tfevents.1731235362.LAPTOP-H8HJ1JCA.13996.0
│   └── model.pt
├── requirements.txt
├── README.md
└── setup.py
```

### Directory and File Descriptions

- **data/**: Directory for data handling and dataset management.
  - **__init__.py**: Marks `data/` as a package.
  - **dataset.py**: Contains classes and methods for loading and preprocessing the text dataset used for training.
  - **evaluate_string.json**: Configuration file for evaluation containing custom evaluation strings.
  - **input.txt**: Tiny Shakespeare dataset used as the training dataset.

- **models/**: Contains the model architectures.
  - **__init__.py**: Marks `models/` as a package.
  - **transformer.py**: Defines the Transformer model architecture with self-attention layers for character-level language modeling.
  - **rnn.py**: Implements the Recurrent Neural Network (RNN) model.
  - **gru.py**: Implements the Gated Recurrent Unit (GRU) model.

- **utils/**: Contains helper functions and utility classes for the project.
  - **__init__.py**: Marks `utils/` as a package.
  - **helpers.py**: Utility functions for data manipulation, logging, and other supportive tasks.
  - **sampling.py**: Defines functions for generating text samples from trained models.

- **scripts/**: Contains executable scripts for training models and generating samples.
  - **__init__.py**: Marks `scripts/` as a package.
  - **train.py**: Script for training a model with specified parameters (e.g., model type, batch size, learning rate).
  - **generate_samples.py**: Script for generating text samples using a pre-trained model.
  - **evaluate_model.py**: Script for evaluating a trained model on PPL, BLEU and ROUGE.

- **tests/**: Contains unit tests to ensure code correctness.
  - **test_models.py**: Unit tests for checking the functionality and correctness of the model architectures.

- **output_directory/**: Stores model checkpoints, event logs, and generated outputs.
  - **model.pt**: Trained model checkpoint.
  - **events.out.tfevents...**: TensorBoard event files for logging training metrics.

- **requirements.txt**: Lists the project dependencies.
- **setup.py**: Installation script for setting up the project.

## Requirements

- Python 3.6 or higher
- PyTorch
- TensorBoard

All dependencies are listed in `requirements.txt`.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your_username/your_project.git
   cd your_project
   ```

2. **Set up a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install project in editable mode (optional)**:

   ```bash
   pip install -e .
   ```

## Usage

### Training

To train a model, use the `train.py` script in the `scripts/` directory.

#### Example Command

```bash
python scripts/train.py \
    --input_file "data/input.txt" \
    --work_dir "output_directory" \
    --type "transformer" \
    --block_size 64 \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 256 \
    --batch_size 64 \
    --num_epochs 10 \
    --learning_rate 0.0001 \
    --device "cuda" \
    --seed 42 \
    --sample_interval 2
```

#### Command-Line Arguments

- `--input-file (-i)`: Path to the input text file.
- `--work-dir (-o)`: Directory to save models and logs.
- `--type`: Model type (options: `transformer`, `rnn`, `gru`, `lstm`).
- `--batch-size (-b)`: Training batch size.
- `--learning-rate (-l)`: Learning rate.
- `--max-steps`: Maximum training steps.
- `--device`: Device for training (`cpu`, `cuda`, etc.).

Additional arguments are available by running:

```bash
python scripts/train.py --help
```

### Generating Samples

After training, generate text samples using the `generate_samples.py` script.

#### Example

```bash
python scripts/generate_samples.py \
    --input_file "data/input.txt" \
    --model_path 'output_directory/model.pt' \
    --device 'cuda' \
    --num_chars 500 \
    --block_size 64 \
    --top_k 40 \
    --start_string "ROMEO: "
```

#### Command-Line Arguments

- `--input-file (-i)`: Input text file (for vocabulary).
- `--model-path (-m)`: Path to the trained model.
- `--num-chars`: Number of characters to generate.
- `--device`: Device for text generation (`cpu`, `cuda`, etc.).

### Evaluation

To evaluate a trained model, use the `evaluate_model.py` script in the `scripts/` directory. This script loads the model and computes evaluation metrics, including BLEU, ROUGE, and Perplexity (PPL), based on a dataset.

#### Example Command

```bash
python scripts/evaluate_model.py \
    --model_path output_directory/model.pt \
    --input_file data/input.txt
```

#### Evaluation Output

The evaluation script calculates and displays the following metrics:
- **BLEU Score**: Measures similarity between generated and target text based on overlapping n-grams.
- **ROUGE Scores (ROUGE-1, ROUGE-2, ROUGE-L)**: Measures text overlap by evaluating precision, recall, and F1 scores for unigrams, bigrams, and longest common subsequences.
- **Perplexity (PPL)**: Evaluates the language model’s performance by analyzing token likelihood.

## TensorBoard Visualization

You can visualize training metrics in real-time using **TensorBoard**. Follow these steps to set up and view the TensorBoard logs:

1. **Install TensorBoard** (if not already installed):

   ```bash
   pip install tensorboard
   ```

2. **Start TensorBoard**: Navigate to the directory containing the `events.out.tfevents` files (typically `output_directory/`), then start the TensorBoard server by running:

   ```bash
   tensorboard --logdir=output_directory
   ```

   Here, `output_directory`

 is the path to the folder containing the TensorBoard log files.

3. **Access TensorBoard**: After starting TensorBoard, you'll see output similar to:

   ```
   Serving TensorBoard on localhost; to access it:
   http://localhost:6006/
   ```

   Open a browser and enter the address (usually `http://localhost:6006`) to view the training metrics such as loss, accuracy, and other relevant data.

Using this approach, you can monitor your training metrics in real-time, which helps in assessing the model's performance and making adjustments to training parameters as needed.

## Configuration

Hyperparameters can be adjusted via command-line arguments. Default values are defined in the scripts.

## Testing

To run unit tests, use:

```bash
python -m unittest discover tests
```

## License

This project is licensed under the MIT License.