import torch
import numpy as np


class InformationParity:

    def __init__(
        self,
        model,
        tokenizer,
        is_sentence_piece_tokenizer: bool = False,
        device=None,
    ):

        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model

        # set model to device
        self.model.to(self.device)
        # set model into eval mode
        self.model.eval()
        self.tokenizer = tokenizer
        self.is_sentence_piece_tokenizer = is_sentence_piece_tokenizer
        self.should_add_bos_token = not self.is_sentence_piece_tokenizer
        self.default_lang_log_loss_cache = {}

    def compute_information_parity(
        self, default_lang_texts: list[str], lang_texts: list[str]
    ):
        """
        Compute information parity across lists of English and non-English texts.

        Args:
            eng_texts: List of English texts
            lang_texts: List of non-English texts

        Returns:
            tuple: (average parity score, standard deviation of parity scores)
        """
        if len(default_lang_texts) != len(lang_texts):
            raise ValueError(
                "The lists of English and non-English texts must have the same length"
            )

        # Compute parity for each pair
        parity_scores = [
            self.compute_pair_information_parity(default_lang_text, lang_text)
            for default_lang_text, lang_text in zip(default_lang_texts, lang_texts)
        ]

        # Calculate average and standard deviation
        avg_parity = np.mean(parity_scores)
        std_parity = np.std(parity_scores)

        return avg_parity, std_parity

    def compute_pair_information_parity(self, default_lang_text: str, lang_text: str):
        # replace the above with get and default
        default_lang_score = self.default_lang_log_loss_cache.get(
            default_lang_text, self.get_text_log_loss(default_lang_text)
        )
        # Cache the score
        self.default_lang_log_loss_cache[default_lang_text] = default_lang_score
        lang_score = self.get_text_log_loss(lang_text)
        return default_lang_score / lang_score

    def get_text_log_loss(
        self,
        text: str,
    ):
        normalized_text = InformationParity._normalize_text(text)
        tokenized_text = self._tokenize_text(normalized_text)
        log_loss, _, _ = self._get_tokens_log_loss(tokenized_text)

        return log_loss.sum().item()

    def _tokenize_text(self, text: str):
        tokenized_text = self.tokenizer(
            self.tokenizer.bos_token + text if self.should_add_bos_token else text,
            return_tensors="pt",
            truncation=True,
        )
        return tokenized_text

    @torch.no_grad()
    def _get_tokens_log_loss(
        self, tokenized_text: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # convert tokenized text to gpu tensors
        tokenized_text = {k: v.to(self.device) for k, v in tokenized_text.items()}
        # Get probabilities from logits
        logits = self.model(**tokenized_text).logits

        relevant_tokens = tokenized_text["input_ids"][0][1:]

        log_loss = InformationParity._calculate_log_loss_from_logits(
            logits, relevant_tokens
        )

        return log_loss, relevant_tokens, logits

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\t", " ")
        return text

    @staticmethod
    def _calculate_log_loss_from_logits(logits, relevant_tokens):
        probs = logits[0].softmax(dim=-1)

        # Get the actual probs of each token
        actual_probs = probs[range(len(probs) - 1), relevant_tokens]

        # Get cross entropy of each token
        log_loss = -actual_probs.log2()
        return log_loss
