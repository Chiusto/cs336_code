from typing import Iterable, Iterator, List, Dict, Tuple, Optional
import ast
import regex
GPT2_PRETOKENIZE_PATTERN = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|"""
        r"""\s+(?!\S)|\s+"""
    )

class Tokenizer:
    def __init__ (self,vocab,merges,special_tokens=None):
        # vocab: dict[int, bytes]
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            special_tokens = []
        self.special_tokens = special_tokens
        #  反向映射
        self.token_to_id = {token:idx for idx,token in vocab.items()}
        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.merges_set = set(self.merges)
        # 处理特殊token
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                idx = len(self.vocab)
                self.vocab[idx] = token_bytes
                self.token_to_id[token_bytes] = idx

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 读取vocab
        vocab ={}
        with open(vocab_filepath,'r',encoding="UTF-8") as f:
            for line in f:
                line =line.strip()
                if not line: continue
                idx, token = line.split(" ", 1)
                # 将字符串 "b'abc'" 转换成 bytes 类型 b'abc'
                vocab[int(idx)] = ast.literal_eval(token)

        # 读取merges
        merges = []
        with open(merges_filepath,'r',encoding="UTF-8") as f:
            for line in f:
                # 移除字符串开头和结尾的空白等字符
                line = line.strip()
                if not line: continue
                token1, token2 = line.split()
                merges.append((
                    ast.literal_eval(token1),
                    ast.literal_eval(token2),
                ))
        # 相当于直接调用该类的 __init__ 构造方法，用这三个参数在内存中实例化出一个新的对象
        return cls(vocab,merges,special_tokens)
    
    def encode(self, text: str) -> list[int]:
        # 最小改动：处理特殊token，保持其不被切分/BPE
        if self.special_tokens:
            sorted_toks = sorted(self.special_tokens, key=len, reverse=True)
            import regex as _re
            pattern = "(" + "|".join(_re.escape(t) for t in sorted_toks) + ")"
            parts = _re.split(pattern, text)
            ids = []
            for part in parts:
                if part in self.special_tokens:
                    ids.append(self.token_to_id[part.encode("utf-8")])
                elif part:
                    for match in _re.finditer(GPT2_PRETOKENIZE_PATTERN, part):
                        token = match.group(0).encode("utf-8")
                        word = tuple(token[i:i+1] for i in range(len(token)))
                        while True:
                            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
                            valid = [p for p in pairs if p in self.merges_set]
                            if not valid:
                                break
                            best_pair = min(valid, key=self.merges_rank.get)
                            word = merge_word(word, best_pair)
                        for b in word:
                            ids.append(self.token_to_id[b])
            return ids
        ids = []
        for match in regex.finditer(GPT2_PRETOKENIZE_PATTERN, text):
            token = match.group(0).encode("utf-8")
            word = tuple(token[i:i+1] for i in range(len(token)))
            while True:
                pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
                valid = [p for p in pairs if p in self.merges_set]
                if not valid: 
                    break
                best_pair = min(valid, key=self.merges_rank.get)
                word = merge_word(word, best_pair)
            for b in word:
                ids.append(self.token_to_id[b])
        return ids 

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        result = b''
        for id in ids:
            result += self.vocab[id]
        return result.decode("UTF-8", errors="replace")

def merge_word(word: tuple[bytes, ...],best_pair: tuple[bytes, bytes],) -> tuple[bytes, ...]:
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
            new_word.append(word[i] + word[i + 1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)