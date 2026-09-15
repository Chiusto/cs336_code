import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件切分成可以独立计数的多个部分。
    如果边界最终发生重叠，则可能返回比指定更少的块数。
    """
    assert isinstance(split_special_token, bytes), "必须将特殊 token 表示为一个字节串"

    # 获取文件的总字节大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 对块边界位置的初始估计，均匀分布
    # 块从上一个索引开始，不包含最后一个索引
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 每次向前读取 4k 字节

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 从边界估计位置开始
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读取一个迷你块

            # 如果到达文件末尾，此边界应位于文件结尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在迷你块中查找特殊 token
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 确保所有边界都是唯一的，但块数可能少于 desired_num_chunks
    return sorted(set(chunk_boundaries))


## 用法
if __name__ == "__main__":
    with open(..., "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # 下面是串行实现，但你可以通过将每个 start/end 对
        # 发送给一组进程来实现并行化。
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # 对你的块进行预分词，并记录每个预 token 的计数
