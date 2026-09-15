import json, ast, re, pathlib, sys
import numpy as np
from tqdm import tqdm
from cs336_basics.tokenizer.tokenizer import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parent
VOCAB_PATH = ROOT / "data" / "tinystories_vocab.json"
MERGES_PATH = ROOT / "data" / "tinystories_merges.txt"

def load_vocab_and_merges():
    print(f"Loading vocab from {VOCAB_PATH}")
    data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    vocab = {int(k): ast.literal_eval(v) for k, v in data.items()}
    print(f"vocab size {len(vocab)}")
    print(f"Loading merges from {MERGES_PATH}")
    # robust parse: find b'...' and b"..."
    pat = re.compile(r"b'(?:[^'\\]|\\.)*'|b\"(?:[^\"\\]|\\.)*\"")
    merges = []
    with open(MERGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            m = pat.findall(line)
            if len(m) >= 2:
                merges.append((ast.literal_eval(m[0]), ast.literal_eval(m[1])))
            else:
                # fallback
                try:
                    a,b = line.split()
                    merges.append((ast.literal_eval(a), ast.literal_eval(b)))
                except: pass
    print(f"merges {len(merges)}")
    return vocab, merges

def encode_file(input_path, output_path, tokenizer, chunk_lines=1000):
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    print(f"\nEncoding {input_path} -> {output_path}")
    total_tokens = 0
    # 先删除旧文件
    if output_path.exists():
        output_path.unlink()
    # 用二进制追加
    with open(input_path, "r", encoding="utf-8", errors="ignore") as fin, open(output_path, "wb") as fout:
        # 统计总行数用于进度（可选）
        # 逐批读取
        batch = []
        pbar = tqdm(unit="lines", desc=input_path.name)
        for line in fin:
            batch.append(line)
            if len(batch) >= chunk_lines:
                ids = []
                for txt in batch:
                    ids.extend(tokenizer.encode(txt))
                if ids:
                    np.array(ids, dtype=np.uint16).tofile(fout)
                    total_tokens += len(ids)
                    pbar.update(len(batch))
                    pbar.set_postfix(tokens=total_tokens)
                batch = []
        # 剩余
        if batch:
            ids = []
            for txt in batch:
                ids.extend(tokenizer.encode(txt))
            if ids:
                np.array(ids, dtype=np.uint16).tofile(fout)
                total_tokens += len(ids)
                pbar.update(len(batch))
        pbar.close()
    print(f"Done {output_path} total_tokens={total_tokens} size={output_path.stat().st_size} bytes")
    # 验证能被 memmap 读取
    arr = np.memmap(str(output_path), dtype=np.uint16, mode="r")
    print(f"  memmap check: len={len(arr)} first10={arr[:10].tolist()}")
    return total_tokens

def main():
    vocab, merges = load_vocab_and_merges()
    tok = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    # quick test
    test = "Hello world! This is a test."
    print(f"test encode: {tok.encode(test)[:20]} decode: {tok.decode(tok.encode(test))[:50]}")
    encode_file(ROOT/"data"/"TinyStoriesV2-GPT4-train.txt", ROOT/"data"/"TinyStories_train.bin", tok)
    encode_file(ROOT/"data"/"TinyStoriesV2-GPT4-valid.txt", ROOT/"data"/"TinyStories_valid.bin", tok)

if __name__ == "__main__":
    main()
