from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def train_tokenizer(
    corpus_files: list[str],
    vocab_size: int = 32000,
    output_path: str = "tokenizer.json",
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train(corpus_files, trainer)
    tokenizer.save(output_path)
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)
