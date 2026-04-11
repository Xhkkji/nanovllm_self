from nanovllm.llm import LLM
from nanovllm.engine.block_manager import block_manager
import time

llm = LLM()
text = "Introduce the acg in China where nearby Japan."
tokens = llm.encoder(text)
all_tokens = list(tokens['input_ids'])
# print(all_tokens)

# 测试
bm = block_manager(num_blocks=10)
# 测试1: 分配
print("\n1. 分配3个块:")
blocks1 = bm.allocate_block(3)
print(f"   分配: {blocks1}")
print(f"   空闲: {list(bm.free_blocks_idx)}")
print(f"   使用: {bm.used_blocks_idx}")


# 测试3: 释放（应该不释放，因为引用计数>0）
print("\n3. 释放第一个块（引用计数2→1）:")
bm.free_block([blocks1[0]])
print(f"   块{blocks1[0]} 引用计数: {bm.blocks[blocks1[0]].ref_count}")
print(f"   空闲块: {list(bm.free_blocks_idx)}")  # 应该没有变化


# 测试5: 释放剩余块
print("\n5. 释放剩余块:")
bm.free_block(blocks1[1:])
print(f"   空闲块: {list(bm.free_blocks_idx)}")
print(f"   使用中: {bm.used_blocks_idx}")

print("\n✅ 所有测试通过!")


# start_time = time.time()
# all_tokens = llm.generate(all_tokens, 500, temperature=0.7)
# end_time = time.time()
# print(f'using time:{end_time - start_time}')
# print(llm.decode(all_tokens))
