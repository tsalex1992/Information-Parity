# Information Parity

A Python package for measuring Information Parity of language models across different languages.

## Overview

Information Parity (IP) is a metric that can predict an LLM's capabilities across multiple languages in a task-agnostic manner. It measures how efficiently language models represent/predict text in different languages relative to a reference language (typically English). It uses cross-entropy loss as a proxy for representation efficiency and calculates a parity score between languages.

From [Information Parity: Measuring and Predicting the Multilingual Capabilities of Language Models](https://aclanthology.org/2024.findings-emnlp.468/):

> We propose a metric called Information Parity (IP) that can predict an LLM's capabilities across multiple languages in a task-agnostic manner. IP is well-motivated from an information theoretic perspective: it is associated with the LLM's efficiency of compressing the text in a given language compared to a reference language.

## Key Features

- **Task-Agnostic Evaluation**: Predicts language model capabilities across languages without requiring task-specific benchmarks
- **Information Theoretical Foundation**: Based on the model's efficiency in predicting text across languages
- **Strong Correlation**: Better correlated with existing task-specific benchmark scores compared to other metrics like Tokenization Parity (TP) and Tokenizer Fertility (TF)
- **Model Ranking**: Useful for ranking multilingual LLM capabilities regardless of the downstream task

## How Information Parity Works

Roughly speaking, for text in language L, IP is the ratio between the English variant of the text's negative log-likelihood and the language L text's negative log-likelihood. The library calculates how efficiently a language model can predict tokens in different languages by:

1. Computing the log loss (cross-entropy) for text in a reference language (usually English)
2. Computing the log loss for equivalent text in another language
3. Calculating the ratio between these values

A parity score of 1.0 indicates equal representation efficiency, while values below 1.0 suggest the model represents the non-reference language less efficiently.

## Installation

Requires Python 3.10+:

```bash
pip install information-parity
```

For running the FLORES-200 evaluation script, you'll need to install the optional dependencies:

```bash
pip install information-parity[evaluation]
```

## Usage

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from information_parity import InformationParity

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Initialize InformationParity
ip = InformationParity(
    model=model,
    tokenizer=tokenizer,
    is_sentence_piece_tokenizer=True
)

# Calculate parity between English and another language
english_text = "This is an example text."
other_text = "Это пример текста."

parity_score = ip.compute_pair_information_parity(english_text, other_text)
print(f"Information parity score: {parity_score}")
```

### Setting the `is_sentence_piece_tokenizer` Flag

The `is_sentence_piece_tokenizer` parameter is crucial for correct tokenization handling:

- Set to `True` for models using SentencePiece tokenizers (like Llama, Gemma, and most modern multilingual models)
- Set to `False` for models using BPE or other tokenization methods

This flag affects how beginning-of-sequence tokens are handled:
- When `True`: The library assumes the tokenizer handles BOS tokens internally
- When `False`: The library explicitly adds BOS tokens to the text

Using the wrong setting can lead to inaccurate parity score calculations. For most recent multilingual LLMs (Llama2, Gemma, Mistral), set this to `True`.

### Evaluating Multiple Text Pairs

```python
english_texts = ["First example", "Second example"]
spanish_texts = ["Primer ejemplo", "Segundo ejemplo"]

avg_parity, std_parity = ip.compute_information_parity(english_texts, spanish_texts)
print(f"Average parity: {avg_parity}, Standard deviation: {std_parity}")
```

### Current Limitations

The current implementation processes text pairs sequentially and doesn't utilize GPU batching. This means that for large datasets, evaluation may take considerable time, especially with larger models.

We welcome contributions to improve performance through:
- Implementing efficient GPU batching
- Parallelizing computations where possible
- Optimizing tokenization and inference processes

## FLORES-200 Evaluation

This repository includes a script [`eval_flores_200.py`](eval_flores_200.py) to evaluate information parity across multiple languages using the FLORES-200 dataset:

```bash
python eval_flores_200.py
```

This will:
1. Load the FLORES-200 dataset
2. Evaluate information parity across all supported languages (using English as reference)
3. Display and save detailed results

## Supported Models

The library has been evaluated on several variants of open-source LLMs:
- Llama2
- Gemma
- Mistral

## Requirements

- Python ≥ 3.10
- transformers ≥ 4.51.2
- torch ≥ 2.0.0
- numpy ≥ 1.24.0
- tqdm ≥ 4.64.1
- datasets ≥ 2.14.0 (optional, for FLORES evaluation)

## Contributing

Contributions are welcome! Areas that would particularly benefit from improvements include:

- GPU batching implementation for faster processing
- Support for additional model architectures
- Improved caching mechanisms
- Optimization of text processing pipelines

To contribute, please open an issue or submit a pull request on the repository.

## Citation

If you use this package in your research, please cite the original paper:

```
@inproceedings{tsvetkov-kipnis-2024-information,
    title = "Information Parity: Measuring and Predicting the Multilingual Capabilities of Language Models",
    author = "Tsvetkov, Alexander  and
      Kipnis, Alon",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.468/",
    doi = "10.18653/v1/2024.findings-emnlp.468",
    pages = "7971--7989",
    abstract = "Large Language Models (LLMs) are increasingly deployed in user-facing applications worldwide, necessitating handling multiple languages across various tasks. We propose a metric called Information Parity (IP) that can predict an LLM`s capabilities across multiple languages in a task-agnostic manner. IP is well-motivated from an information theoretic perspective: it is associated with the LLM`s efficiency of compressing the text in a given language compared to a reference language. We evaluate IP and other popular metrics such as Tokenization Parity (TP) and Tokenizer Fertility (TF) on several variants of open-sourced LLMs (Llama2, Gemma, Mistral). Among all metrics known to us, IP is better correlated with existing task-specific benchmark scores from the literature and thus better predicts such scores in a certain language. These findings show that IP may be useful for ranking multilingual LLMs' capabilities regardless of the downstream task."
}
```

## License

MIT License