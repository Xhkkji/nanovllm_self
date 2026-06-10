from collections import deque
from typing import List, Tuple
import xxhash
import torch

from .Sequence import Sequence

engine_config = {
    "enable_prefix_caching": True,  # 启用前缀共享
    "prefix_cache_hash_algo": "sha256",  # 哈希算法
    "block_size": 32,  # 推荐 32
}

class Block:
    """
    用于维护kvcache的状态，真正的存储在kvcache中，kvcache依靠block索引与block联系
    """
    def __init__(self, block_id, block_size):
        self.block_id = block_id  # 分配好就不再修改
        self.block_size = block_size  # 分配好就不再修改
        self.ref_count = 0
        self.hash = -1  # 只有当前块满的时候才会根据prev计算出当前块的hash
        self.token_ids = []
        self.ptr = None  # 指向GPU物理块的指针
        self.prev_hash = -1  # 除了第一个块以外一定有一个值，即上一个block的hash

    def update(self, h: int, token_ids: List[int], prev_hash):
        self.hash = h
        self.token_ids = token_ids
        self.prev_hash = prev_hash

    def reset(self):
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        self.prev_hash = -1

class block_manager:
    def __init__(self, num_blocks=1024, block_size=32, num_layers=32, num_kv_heads=32, head_dim=128, enable_prefix_cache=False, dtype=torch.bfloat16, device='cuda:0'):
        """
        总共 4096 个块
        每个块4个token
        """
        self.num_blocks = num_blocks  # 需要分配的块的数量
        self.block_size = block_size  # 一个块分配多大的空间，即一个块多少个token
        self.enable_prefix_cache = enable_prefix_cache  # 启用前缀共享
        self.blocks = [Block(i, block_size) for i in range(num_blocks)]  # 根据设置的块数量分配块
        self.hash_to_block_id:dict[int, int] = dict()  # 链式哈希对应的块，都是已满的块
        self.free_blocks_idx:deque[int] = deque(range(num_blocks))  # 生成1-num_blocks的整数序列，表示哪些块还未分配
        self.used_blocks_idx:set[int] = set()  # 哪些块已被使用

        # # 分配kv_cache
        # # 形状: [num_blocks, block_size, 2(key和value), num_layers, num_kv_heads, head_dim]
        # self.kv_cache = torch.zeros(
        #     num_blocks, block_size, 2, num_layers, num_kv_heads, head_dim,
        #     dtype=dtype, device=device
        # )
    
    def can_allocate(self, seq:Sequence):
        """
        prefill阶段判断剩余空间够不够装下一整个seq
        """
        if (len(seq) + self.block_size - 1) // self.block_size > len(self.free_blocks_idx):
            return False
        else:
            return True
    
    def can_append(self, seq:Sequence):
        """
        新token还未生成，只是检查预分配空间
        检查是否够空间，不改变任何内部状态
        decode阶段判断剩余空间够不够装下一个token,但是要考虑已有block未装满的情况
        """
        # 判断“是否将要或刚刚进入一个新的物理块
        # 并且即使没有free_block了也不一定就不能append，因为可能有未满的已分配块
        if len(seq) % self.block_size == 1:
            return len(self.free_blocks_idx) >= 1
        return True
    
    def may_append(self, seq:Sequence):
        """
        新token还未生成，只是预分配空间
        实际分配block
        """
        # 由于是预分配，当余数为1时意味着新token恰好需要分配新块，为0意味着新token进入可以刚好装下
        # 如果余数是2、3等，直接加入到未满的块即可，不需要新分配块
        if len(seq) % self.block_size == 1:  # 判断“是否将要或刚刚进入一个新的物理块
            new_block_id = self.allocate_block(1)[0]  # 引用计数已+1
            last_block_id = seq.block_table[-1]
            self.blocks[new_block_id].prev_hash = self.blocks[last_block_id].hash
            seq.block_table.append(new_block_id)
        elif len(seq) % self.block_size == 0:  # 装入后刚好块满，需要更新hash
            last_block_id = seq.block_table[-1]
            token_ids = seq.token_ids[-self.block_size:]
            prefix = self.blocks[seq.block_table[-2]].hash if len(seq.block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            self.blocks[last_block_id].update(h, token_ids, prefix)
            self.hash_to_block_id[h] = last_block_id
        else:
            last_block_id = seq.block_table[-1]
            assert self.blocks[last_block_id].hash == -1  # 块未满，hash不赋值
    
    def allocate(self, seq:Sequence):
        return self.allocate_with_prefill(seq)
    
    def deallocate(self, seq:Sequence):
        if not seq.block_table:
            return
        
        block_ids = seq.block_table
        self.free_blocks(block_ids)
        seq.block_table = []
        seq.num_cached_tokens = 0

    # def set_kv(self, block_id, offset, layer, k, v):
    #     """写入 KV 到指定块的位置， offset为块内偏移，即第几个token"""
    #     # k, v 形状: [num_kv_heads, head_dim]
    #     self.kv_cache[block_id, offset, 0, layer] = k
    #     self.kv_cache[block_id, offset, 1, layer] = v
    
    # def set_kv_prefill(self, k, v, seq, layer):
    #     """
    #     prefill阶段一次性存储所有token的kv
    #     k, v 形状: [seq_len, num_kv_heads, head_dim]
    #     """
    #     # seq.token_ids.shape=torch.Size([1, token_len])
    #     # print(f"seq.token_ids:{seq.token_ids}")
    #     token_ids = seq.token_ids  # 取出 token 列表
    #     block_table = seq.block_table

    #     for token_idx in range(len(token_ids)):
    #         block_idx = token_idx // self.block_size
    #         block_id = block_table[block_idx]  # 取出对应的真实物理块id
    #         offset = token_idx % self.block_size
    #         self.set_kv(block_id, offset, layer, k[token_idx], v[token_idx])


    # def get_kv(self, block_id, offset, layer):
    #     """从指定块位置取出已经计算好的k和v,offset为块内偏移，即第几个token"""
    #     # k, v 形状: [num_kv_heads, head_dim]
    #     k = self.kv_cache[block_id, offset, 0, layer]
    #     v = self.kv_cache[block_id, offset, 1, layer]
    #     return k, v
    
    # def get_kv_block(self, seq, layer):
    #     """
    #     每次decode过程直接成批地取出上文的kv
    #     kvcache形状：[num_blocks, block_size, 2, num_layers, num_kv_heads, head_dim]
    #     """
    #     block_size = self.block_size
    #     block_table = seq.block_table
    #     num_tokens = len(seq.token_ids)  # 该seq的token的总数

    #     # print(f'block_table:{block_table}')
    #     # 收集所有的kv
    #     k_blocks = self.kv_cache[block_table, :, 0, layer, :, :]
    #     # [num_seq, block_size, num_kv_heads, head_dim] -> [seq_len, num_kv_heads, head_dim]
    #     all_k = k_blocks.reshape(-1, self.kv_cache.shape[-2], self.kv_cache.shape[-1])
    #     # 截去未满块的空白内容
    #     all_k = all_k[:num_tokens]
    #     v_blocks = self.kv_cache[block_table, :, 1, layer, :, :]
    #     all_v = v_blocks.reshape(-1, self.kv_cache.shape[-2], self.kv_cache.shape[-1])
    #     all_v = all_v[:num_tokens]
    #     # print(f"all_k:{all_k}")
    #     # print(f"all_v:{all_v}")
    #     return all_k, all_v

    # 按批分配，没有计算hash的过程，为block_manager初始化分配时使用的接口
    def allocate_block(self, num_blocks):
        assert num_blocks <= len(self.free_blocks_idx)
        allocate_idx = []  # 储存id
        for i in range(num_blocks):
            block_idx = self.free_blocks_idx.popleft()  #移除
            self.used_blocks_idx.add(block_idx)
            self.blocks[block_idx].ref_count = 1
            allocate_idx.append(block_idx)
        return allocate_idx

    # 根据传入的block_table批量释放block
    def free_blocks(self, block_ids):
        """
        block_ids表示需要释放的块的索引，为链表
        """
        for idx in block_ids:
            assert self.num_blocks > idx >= 0
            assert idx in self.used_blocks_idx
            block_hash = self.blocks[idx].hash
            self.blocks[idx].ref_count -= 1
            if self.blocks[idx].ref_count == 0:
                # 释放哈希列表
                if block_hash != -1 and self.hash_to_block_id.get(block_hash) == idx:
                    self.hash_to_block_id.pop(block_hash, None)

                self.blocks[idx].reset()
                self.free_blocks_idx.appendleft(idx)
                self.used_blocks_idx.discard(idx)
            

    def compute_hash(self, token_ids, prev_hash=-1):
        """
        依据新传入的token序列更新链式hash，针对的是一个token_ids
        """
        h = xxhash.xxh64()  # 哈希对象
        if prev_hash != -1:
            # 一开始的8字节是为了防止对已有的prev_hash进行截断
            h.update(prev_hash.to_bytes(8, "little"))  # 把整数转换为 8 字节的二进制数据, "little"：小端序(Little Endian)
        for idx in token_ids:
            # 对于新加入的每个token，4字节足矣
            h.update(idx.to_bytes(4, "little"))
        return h.intdigest()

    def allocate_with_prefill(self, seq: Sequence):
        """
        传入的是一个seq类，在prefill阶段直接分配块
        复用block，查看前缀block，一旦发生cache_miss,则后续全部分配新block
        """
        prev_hash = -1
        cache_miss = False
        physical_blocks = []
        
        token_ids = seq.token_ids  # 取出 token 列表
        # token_ids 是 List[int]，形状是 [seq_len]
        # 例如: [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # print(f"[Debug] allocate_with_prefill: Allocating blocks for tokens: {token_ids}")  # 添加调试输出，查看输入的 token 列表
        # 计算前缀数组，一旦cachemiss则后续全部新分配
        for i in range(0, len(token_ids), self.block_size):
            block_tokens = token_ids[i: i+self.block_size]  # 以block_size为单位进行切片，若最后一个块不满，会自动切片，只填入剩余的idx
            is_full = len(block_tokens) == self.block_size
            if not is_full:
                current_hash = -1
            else:
                # 以块的token为单位进行更新
                current_hash = self.compute_hash(block_tokens, prev_hash)

            # 尝试复用已分配的block
            # 先从哈希表获取，观察是否命中已有block(特殊判断：若命中，查看储存的token是否真的一致)
            block_id = self.hash_to_block_id.get(current_hash, -1)
            if block_id == -1 or self.blocks[block_id].token_ids!= block_tokens:
                cache_miss = True

            if cache_miss:
                new_idx = self.free_blocks_idx.popleft()
                self.used_blocks_idx.add(new_idx)
                physical_blocks.append(new_idx)
                self.blocks[new_idx].ref_count = 1
                # 更新块状态
                self.blocks[new_idx].update(current_hash, block_tokens, prev_hash)

                # 如果是完整块，注册到哈希表
                if is_full:
                    self.blocks[new_idx].ref_count = 1
                    self.hash_to_block_id[current_hash] = new_idx
            # 复用已有块
            else:
                seq.num_cached_tokens += self.block_size
                self.blocks[block_id].ref_count += 1  # 根据block_id复用已有的块
                physical_blocks.append(block_id)
            # 若当前块已满，更新prev_hash，用于下一轮计算
            if is_full:
                prev_hash = current_hash

        # 记录当前序列分配到的所有物理块 ID
        seq.block_table = physical_blocks
        return physical_blocks



