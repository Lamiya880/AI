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
                if i >= 10000:
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

    print(f"Tokenizing {args.num_samples} samples in batches...")

    import numpy as np
    import torch

    # Pre-allocate numpy memmap on disk (avoids holding everything in RAM)
    memmap_path = output_path.parent / "tokens_memmap.dat"
    estimated_tokens = args.num_samples * 200  # rough estimate: ~200 tokens per sample
    memmap = np.memmap(str(memmap_path), dtype=np.int32, mode="w+", shape=(estimated_tokens,))

    batch_size = 500
    total_tokens = 0
    batch_texts = []
    samples_processed = 0

    for item in ds:
        if samples_processed >= args.num_samples:
            break

        batch_texts.append(item["text"])
        samples_processed += 1

        if len(batch_texts) >= batch_size:
            # Batch tokenize
            encoded = tokenizer.encode_batch(batch_texts)
            for enc in encoded:
                tokens = enc.ids
                end = total_tokens + len(tokens)
                if end > len(memmap):
                    # Extend memmap
                    new_size = max(end, len(memmap) * 2)
                    memmap = np.memmap(str(memmap_path), dtype=np.int32, mode="r+",
                                       shape=(new_size,))
                    # Re-open with new size
                    del memmap
                    memmap = np.memmap(str(memmap_path), dtype=np.int32, mode="r+",
                                       shape=(new_size,))
                memmap[total_tokens:end] = tokens[:len(memmap) - total_tokens]
                total_tokens += len(tokens)

            batch_texts = []
            if samples_processed % 10000 == 0:
                print(f"  Processed {samples_processed:,} samples, {total_tokens:,} tokens")

    # Process remaining texts
    if batch_texts:
        encoded = tokenizer.encode_batch(batch_texts)
        for enc in encoded:
            tokens = enc.ids
            end = total_tokens + len(tokens)
            if end <= len(memmap):
                memmap[total_tokens:end] = tokens
                total_tokens += len(tokens)

    memmap.flush()
    print(f"Tokenized {total_tokens:,} tokens total")

    # Convert to torch tensor and save
    print("Converting to PyTorch tensor...")
    tokens_array = np.memmap(str(memmap_path), dtype=np.int32, mode="r", shape=(total_tokens,))
    tokens_tensor = torch.from_numpy(tokens_array.astype(np.int64))
    torch.save(tokens_tensor, str(output_path))

    # Cleanup memmap file
    memmap_path.unlink(missing_ok=True)

    print(f"Saved {total_tokens:,} tokens to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
