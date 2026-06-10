import torch
import torch.nn as nn


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits, temperatures):
        """
        需要[last_token_of_seq0,
            last_token_of_seq1,
            ...]
            即[num_seqs, vocab_size]
        输入：
        prefill时，logits为: [total_new_tokens, vocab_size]，是展平状态

        logits:
        qwen3里面对prefill处理后
        prefill:[seq_num, vocab_size]
        decode:[seq_num, vocab_size]
        temperature:[seq_num]

        输出：
        prefill:
        torch.list[seq_num]
        decode:
        torch.list[seq_num]
        长度和 seqs 一样
        每个 int 对应一条 seq 的 next token
        """
        output_logits = logits
        if torch.any(temperatures <= 0):
            output_logits = logits
            next_tokens = torch.argmax(output_logits, dim=-1)
        else:
            output_logits = logits.float() / temperatures.unsqueeze(1)
            probs = torch.softmax(output_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
        if next_tokens.dim() == 2:
            next_tokens = next_tokens.squeeze(-1)
            
        return next_tokens