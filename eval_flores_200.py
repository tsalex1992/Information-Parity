from datasets import load_dataset
from information_parity.information_parity import InformationParity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
import json
import datetime

# Define language mappings
lang_code_to_flores_key = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "fr": "fra_Latn",
    "ko": "kor_Hang",
    "ja": "jpn_Jpan",
    "he": "heb_Hebr",
    "hu": "hun_Latn",
    "no": "nob_Latn",
    "hi": "hin_Deva",
    "fi": "fin_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "zh": "zho_Hans",
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "ro": "ron_Latn",
    "uk": "ukr_Cyrl",
    "sr": "srp_Cyrl",
    "hr": "hrv_Latn",
    "da": "dan_Latn",
    "ca": "cat_Latn",
    "ar": "arb_Arab",
    "tr": "tur_Latn",
    "cs": "ces_Latn",
    "th": "tha_Thai",
    "bn": "ben_Beng",
    "bg": "bul_Cyrl",
    "el": "ell_Grek",
    "ur": "urd_Arab",
    "mr": "mar_Deva",
    "eu": "eus_Latn",
    "et": "est_Latn",
    "ms": "zsm_Latn",
    "as": "asm_Beng",
    "gu": "guj_Gujr",
    "ka": "kat_Geor",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "np": "npi_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "my": "mya_Mymr",
    "sw": "swh_Latn",
    "pt": "por_Latn",
    "ht": "hat_Latn",
    "qu": "quy_Latn",
}


# Function to evaluate information parity for a batch of sentences
def evaluate_information_parity(dataset, ip, languages_to_evaluate):
    dict_keys = languages_to_evaluate.keys()
    key_prefix = "sentence_"
    # All languages except English (which is our reference language)
    dict_keys_without_english = [key for key in dict_keys if key != "en"]

    # Format keys for dataset access
    english_key = f"{key_prefix}{languages_to_evaluate['en']}"

    results = {key: [] for key in dict_keys_without_english}

    # Simple loop over all examples in the dataset
    for i in tqdm(range(len(dataset))):
        english_text = dataset[i][english_key]

        for lang_code in dict_keys_without_english:
            lang_key = f"{key_prefix}{languages_to_evaluate[lang_code]}"
            lang_text = dataset[i][lang_key]
            parity_score = ip.compute_pair_information_parity(english_text, lang_text)
            results[lang_code].append(parity_score)

    return results


def main():
    print("Loading FLORES-200 dataset...")
    dataset = load_dataset("facebook/flores", "all")
    dev_test = dataset["devtest"]

    # Choose which languages to evaluate
    # To evaluate more languages, uncomment and modify the following line:
    languages_to_evaluate = {k: v for k, v in lang_code_to_flores_key.items()}

    print(
        f"Will evaluate the following languages: {list(languages_to_evaluate.keys())}"
    )

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    access_token = ""  # Add your token if needed

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", token=access_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        token=access_token,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Initialize the InformationParity evaluator
    ip = InformationParity(
        model=model,
        tokenizer=tokenizer,
        is_sentence_piece_tokenizer=True,
    )
    print(ip.is_sentence_piece_tokenizer)

    # Run evaluation
    print("Evaluating information parity across languages...")
    parity_scores = evaluate_information_parity(dev_test, ip, languages_to_evaluate)

    # Calculate and display statistics
    dict_keys_without_english = [
        key for key in languages_to_evaluate.keys() if key != "en"
    ]

    print("\nInformation Parity Results (English as reference):")
    print("-" * 50)
    print(f"{'Language':10} | {'Average':10} | {'Std Dev':10} | {'Median':10}")
    print("-" * 50)

    for lang_code in dict_keys_without_english:
        scores = parity_scores[lang_code]
        avg_score = np.mean(scores)
        std_dev = np.std(scores)
        median = np.median(scores)
        print(f"{lang_code:10} | {avg_score:.4f}    | {std_dev:.4f}    | {median:.4f}")

    # Save results to file
    results_data = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "results": {
            lang: {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "median": float(np.median(scores)),
                "raw_scores": [float(score) for score in scores],
            }
            for lang, scores in parity_scores.items()
        },
    }

    with open("flores_information_parity_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"Results saved to flores_information_parity_results.json")


if __name__ == "__main__":
    main()
