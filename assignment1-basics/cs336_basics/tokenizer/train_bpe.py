from __future__ import annotations

import heapq
import multiprocessing
import os

import regex

from cs336_basics.tokenizer.pretokenization import find_chunk_boundaries

GPT2_PRETOKENIZE_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|"""
    r"""\s+(?!\S)|\s+"""
)


class _PairHeapEntry:
    def __init__(self, count: int, pair: tuple[bytes, bytes]):
        self.count = count
        self.pair = pair

    def __lt__(self, other: _PairHeapEntry) -> bool:
        # heapq 是最小堆；反转比较以实现最大堆。
        # 同频时与原 max(..., key=lambda x: (x[1], x[0])) 保持一致。
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


# 必须单独作为函数，才能使用 multiprocessing。
def _pretokenize_chunk(args: tuple[str, int, int, tuple[str, ...]]):
    input_path, start, end, special_tokens = args
    word_freq: dict[tuple[bytes, ...], int] = {}
    pair_counts: dict[tuple[bytes, bytes], int] = {}

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # 不要按每次出现的 special token 重复扫描整个 chunk。
    for special_token in special_tokens:
        chunk = chunk.replace(special_token, "")

    for match in regex.finditer(GPT2_PRETOKENIZE_PATTERN, chunk):
        token_bytes = match.group(0).encode("utf-8")
        token_bytes = tuple(token_bytes[i:i + 1] for i in range(len(token_bytes)))

        word_freq[token_bytes] = word_freq.get(token_bytes, 0) + 1

        for i in range(len(token_bytes) - 1):
            pair = (token_bytes[i], token_bytes[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    return word_freq, pair_counts


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}

    for i in range(256):
        vocab[i] = bytes([i])

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    word_freq: dict[tuple[bytes, ...], int] = {}

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    jobs = [
        (input_path, start, end, tuple(special_tokens))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(_pretokenize_chunk, jobs)

    for part_word_freq, part_pair_counts in results:
        for word, count in part_word_freq.items():
            word_freq[word] = word_freq.get(word, 0) + count

        for pair, count in part_pair_counts.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + count

    words_with_pair: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for word in word_freq:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            words_with_pair.setdefault(pair, set()).add(word)

    # 新增：堆使最高频 pair 的选择不必每轮扫描全部 pair_counts。
    pair_heap = [_PairHeapEntry(count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)

    next_id = len(vocab)

    while len(vocab) < vocab_size and pair_counts:
        # 跳过频率更新前残留在堆中的旧记录。
        while pair_heap:
            entry = heapq.heappop(pair_heap)
            best_pair = entry.pair
            frequency = entry.count

            if pair_counts.get(best_pair) == frequency:
                break
        else:
            break

        vocab[next_id] = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        next_id += 1

        affected = list(words_with_pair.get(best_pair, ()))
        word_delta: dict[tuple[bytes, ...], int] = {}
        pair_delta: dict[tuple[bytes, bytes], int] = {}

        for word in affected:
            freq = word_freq[word]
            new_word = merge_word(word, best_pair)

            word_delta[word] = word_delta.get(word, 0) - freq
            word_delta[new_word] = word_delta.get(new_word, 0) + freq

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_delta[pair] = pair_delta.get(pair, 0) - freq

            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                pair_delta[pair] = pair_delta.get(pair, 0) + freq

        for word, delta in word_delta.items():
            count = word_freq.get(word, 0) + delta

            if count > 0:
                word_freq[word] = count
            else:
                word_freq.pop(word, None)

        for pair, delta in pair_delta.items():
            count = pair_counts.get(pair, 0) + delta

            if count > 0:
                pair_counts[pair] = count

                # 新频率入堆；旧频率记录在下次弹出时自动跳过。
                heapq.heappush(pair_heap, _PairHeapEntry(count, pair))
            else:
                pair_counts.pop(pair, None)

        # 保留你原有的反向索引维护逻辑。
        for word in word_delta:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                word_set = words_with_pair.get(pair)

                if word_set is None:
                    word_set = set()
                    words_with_pair[pair] = word_set

                if word_freq.get(word, 0) > 0:
                    word_set.add(word)
                else:
                    word_set.discard(word)

                if not word_set:
                    del words_with_pair[pair]

    return vocab, merges


def merge_word(
    word: tuple[bytes, ...],
    best_pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
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

