import torch
from transformers import PreTrainedTokenizer

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    
    batch_size = len(prompt_strs)
    prompts = [
        tokenizer.encode(x,add_special_tokens=False)
        for x in prompt_strs
    ]
    outputs = [
        tokenizer.encode(x,add_special_tokens=False)
        for x in output_strs
    ]
    sequences = [p+o for p,o in zip(prompts,outputs)]
    max_len = max(len(x) for x in sequences)
    input_ids = torch.full(
        (batch_size,max_len),
        tokenizer.pad_token_id,
        dtype = torch.long,
    )
    labels = torch.full_like(
        input_ids,
        tokenizer.pad_token_id
    )
    for i,seq in enumerate(sequences):
        input_ids[i,:len(seq)] = torch.tensor(seq)
        labels[i,:len(seq)] = torch.tensor(seq)
    input_ids = input_ids[:, :-1]
    labels = labels[:, 1:]

    response_mask = torch.zeros_like(input_ids)
    for i,(p,o) in enumerate(zip(prompts,outputs)):
          start = len(p) - 1
          response_mask[i, start:start+len(o)] = 1

    return {
        "input_ids":input_ids,
        "labels":labels,
        "response_mask":response_mask,
    }