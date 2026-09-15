from pathlib import Path
from cs336_basics.tokenizer.train_bpe import train_bpe
import json

def main():
    ROOT = Path(__file__).resolve().parents[1]
    train_path = ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
    vocab_path = ROOT / "data" / "tinystories_vocab.json"
    merges_path = ROOT / "data" / "tinystories_merges.txt"
    print(f"input: {train_path} exists={train_path.exists()}")
    vocab, merges = train_bpe(input_path=str(train_path), vocab_size=10000, special_tokens=["<|endoftext|>"])
    with open(vocab_path,"w", encoding="utf-8") as f:
        json.dump({str(k): repr(v) for k,v in vocab.items()}, f, ensure_ascii=False)
    with open(merges_path,"w", encoding="utf-8") as f:
        for a,b in merges:
            f.write(f"{repr(a)} {repr(b)}\n")
    print(f"done vocab={len(vocab)} merges={len(merges)}")

if __name__ == "__main__":
    main()
