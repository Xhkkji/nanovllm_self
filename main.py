from nanovllm.llm import LLM, LLM_self
from nanovllm.engine.block_manager import block_manager
import time
import torch
from transformers import AutoConfig

text1 = "What is a large language model?"
text2 = "How does a transformer model work?"

use_self = True
use_multi_seq_test = True
if use_self == False:
    llm = LLM()
    tokens = llm.encoder(text1)
    all_tokens = list(tokens['input_ids'])
    print(all_tokens)
    input_tensor = torch.tensor(all_tokens).unsqueeze(0) # 添加 batch 维度

    start_time = time.time()
    all_tokens = llm.generate(input_tensor, 256, temperature=0.7)
    end_time = time.time()
    print(f'using time:{end_time - start_time}')
    print(llm.decode(all_tokens))
else:
    llm = LLM_self(enable_profile=True)

    if use_multi_seq_test:
        print(f'use_multi_seq_test..')
        tokens1 = llm.encoder(text1)
        tokens2 = llm.encoder(text2)

        ids1 = list(tokens1['input_ids'])
        ids2 = list(tokens2['input_ids'])
        print("seq1 tokens:", ids1)
        print("seq2 tokens:", ids2)

        # 方案A：只测试等长 prompt，避免 padding 干扰当前多 seq 验证
        assert len(ids1) == len(ids2), f"two prompts must have the same token length, got {len(ids1)} and {len(ids2)}"
        input_tensor = torch.tensor([ids1, ids2])
    else:
        tokens1 = llm.encoder(text1)
        all_tokens = list(tokens1['input_ids'])
        print(all_tokens)
        input_tensor = torch.tensor(all_tokens).unsqueeze(0)  # 保持你原来的单条逻辑

    print(f"input_tensor shape: {input_tensor.shape}")  # 添加调试输出，查看输入张量的形状
    start_time = time.time()
    all_tokens = llm.generate(input_tensor)
    end_time = time.time()
    print(f'using time:{end_time - start_time}')
    if use_multi_seq_test:
        for tokens in all_tokens:
            print(llm.decode(tokens))
    else:
        print(llm.decode(all_tokens))



