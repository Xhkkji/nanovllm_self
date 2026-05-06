from nanovllm.llm import LLM, LLM_self
from nanovllm.engine.block_manager import block_manager
import time
import torch
from transformers import AutoConfig

# llm = LLM()
# text = "Introduce the acg in China where nearby Japan."
# tokens = llm.encoder(text)
# all_tokens = list(tokens['input_ids'])
# print(all_tokens)
# input_tensor = torch.tensor(all_tokens).unsqueeze(0) # 添加 batch 维度

# start_time = time.time()
# all_tokens = llm.generate(input_tensor, 500, temperature=0.7)
# end_time = time.time()
# print(f'using time:{end_time - start_time}')
# print(llm.decode(all_tokens))

llm = LLM_self()
# text = "Introduce the acg in China where nearby Japan."
text = "请用自然、生动、带有画面感的语言，介绍中国的ACG文化，并简要对比日本ACG文化。"


tokens = llm.encoder(text)
all_tokens = list(tokens['input_ids'])
print(all_tokens)
input_tensor = torch.tensor(all_tokens).unsqueeze(0)  # 为了方便批处理，batch维度最好保留
print(f"input_tensor shape: {input_tensor.shape}")  # 添加调试输出，查看输入张量的形状
start_time = time.time()
all_tokens = llm.generate(input_tensor, 500, temperature=1)
end_time = time.time()
print(f'using time:{end_time - start_time}')
print(llm.decode(all_tokens))





