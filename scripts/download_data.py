import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download and tokenize training data")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", type=str, default="sample-10BT")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--output", type=str, default="data/train_tokens.pt")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer.json")
    parser.add_argument("--train-tokenizer", action="store_true", help="Train a new tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    print(f"Loading dataset {args.dataset}/{args.subset}...")
    ds = load_dataset(args.dataset, args.subset, split=args.split, streaming=True)

    if args.train_tokenizer:
        print(f"Training tokenizer with vocab size {args.vocab_size}...")
        from training.tokenizer import train_tokenizer

        sample_file = output_path.parent / "sample_text.txt"
        with open(sample_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(ds):
                if i >= 50000:
                    break
                f.write(item["text"] + "\n")

        train_tokenizer([str(sample_file)], args.vocab_size, args.tokenizer_path)
        print(f"Tokenizer saved to {args.tokenizer_path}")

    from training.tokenizer import load_tokenizer

    if not Path(args.tokenizer_path).exists():
        print(f"Tokenizer not found at {args.tokenizer_path}. Use --train-tokenizer to create one.")
        sys.exit(1)

    tokenizer = load_tokenizer(args.tokenizer_path)
    ds = load_dataset(args.dataset, args.subset, split=args.split, streaming=True)

    print(f"Tokenizing {args.num_samples} samples...")
    all_tokens = []
    for i, item in enumerate(ds):
        if i >= args.num_samples:
            break
        tokens = tokenizer.encode(item["text"]).ids
        all_tokens.extend(tokens)
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1} samples, {len(all_tokens):,} tokens")

    import torch
    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
    torch.save(tokens_tensor, str(output_path))

    print(f"Saved {len(all_tokens):,} tokens to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
