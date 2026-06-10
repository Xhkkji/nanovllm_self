import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from nanovllm.config import Config
from ..models.qwen3 import Qwen3Model
from transformers import AutoConfig
from nanovllm.engine.block_manager import block_manager
from nanovllm.engine.Sequence import Sequence
from nanovllm.utils.context import Context, get_context
from nanovllm.layers.sampler import Sampler

class ModelRunner(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.device = self.config.device
        model_config = AutoConfig.from_pretrained("/home/xhk/model/Qwen3-0.6B/")
        self.model = Qwen3Model(model_config).to(self.device)
        self.sampler = Sampler()
        print("\n创建 BlockManager...")
        self.block_manager = block_manager(
            num_blocks=self.config.num_blocks,
            block_size=self.config.block_size,
            num_layers=self.model.num_layers,
            num_kv_heads=self.model.num_kv_heads,
            head_dim=self.model.head_dim
        )
        print("✅ BlockManager 创建成功")
        # 分配kv_cache
        # 形状: [2(key和value), num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_cache = torch.zeros(
            2, 
            self.model.num_layers, 
            self.config.num_blocks,
            self.config.block_size, 
            self.model.num_kv_heads, 
            self.model.head_dim,
            dtype=config.dtype, 
            device=config.device
        )
        self.bind_kvcache_to_attention()
    
    def bind_kvcache_to_attention(self):
        """
        把全局 kv_cache 中每一层对应的视图，提前绑定到该层的 PagedAttention 上。
        这样前向时就不用每次手动传全局 cache。
        """
        for layer_id, layer in enumerate(self.model.layers):
            # 取出对应qwendecodelayer中的p_attn
            p_attn = layer.p_attn

            p_attn.k_cache = self.kv_cache[0, layer_id]
            p_attn.v_cache = self.kv_cache[1, layer_id]
            p_attn.block_size = self.config.block_size
            p_attn.layer_id = layer_id

    
    def prepare_block_tables(self, seqs):
        """
        数据预处理和传输的过程：先将变长的块表填充成规整的矩阵
        再把这个“地址表”高效地传输给 GPU，供 Attention 内核快速查找 KV 缓存的位置。
        block_tables[i][j]: 第 i 条序列的第 j 个逻辑块，映射到哪个物理块
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # pin_memory固定一块cpu内存，可以加速cpu到gpu的传输，noBlock为异步传输
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).to(self.config.device)
        return block_tables
    
    def prepare_sampler(self, seqs:list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).to(self.config.device)
        return temperatures


    def prepare_prefill(self, seqs: list[Sequence]):
        """
        seqA.token_ids = [11, 12, 13, 14, 15]
        seqB.token_ids = [21, 22, 23]
        打包成
        input_ids = [11, 12, 13, 14, 15, 21, 22, 23]
        positions = [0, 1, 2, 3, 4, 0, 1, 2]
        """
        input_ids = []
        positions = []
        # 构建cu_seqlen
        cu_seqlen_q = [0]
        cu_seqlen_k = [0]
        slot_mapping = []
        max_seqlen_q = 0
        max_seqlen_k = 0
        block_tables = None
        
        for seq in seqs:
            seq_len = len(seq)
            # 储存本次需要计算的新token数
            seqlen_q = seq_len - seq.num_cached_tokens
            # KV 缓存的总长度（包括历史和新 token）
            seqlen_k = seq_len

            new_tokens = seq.token_ids[seq.num_cached_tokens:]
            new_positions = list(range(seq.num_cached_tokens, seq_len))

            input_ids.extend(new_tokens)
            positions.extend(new_positions)

            cu_seqlen_q.append(cu_seqlen_q[-1] + seqlen_q)
            cu_seqlen_k.append(cu_seqlen_k[-1] + seqlen_k)
            # 计算最大值
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)  # 相同
            # 遍历每个seq中的所有token，储存slop_map，块映射
            # 利用seq中已经储存的block信息，映射到context的slop_map上下文中
            for token_pos in range(seq.num_cached_tokens, seq_len):
                block_idx = token_pos // seq.block_size
                offset = token_pos % seq.block_size
                block_id = seq.block_table[block_idx]
                slot_mapping.append(block_id * seq.block_size + offset)  # 直接把所有seq的token拉成一列储存

        # cu_seqlens_q[-1]：本轮新算的 query token 总数
        # cu_seqlens_k[-1]：本轮可见的 key/value 总长度
        # 本轮 query 比可见上下文短，说明有部分token不是新算的，命中prefix，在cache里
        
        if cu_seqlen_k[-1] > cu_seqlen_q[-1]:
            # 这里block_table已经在GPU上了
            block_tables = self.prepare_block_tables(seqs)  # 把多个seq长度不相等的blocktable处理成等长的二维矩阵，方便查找

        context = get_context(is_prefill=True, 
            cu_seqlens_q=torch.tensor(cu_seqlen_q, dtype=torch.int32, device=self.device), 
            cu_seqlens_k=torch.tensor(cu_seqlen_k, dtype=torch.int32, device=self.device), 
            max_seqlen_q=max_seqlen_q, 
            max_seqlen_k=max_seqlen_k,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, device=self.device), 
            context_lens=None,
            block_tables=block_tables
            )

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        positions = torch.tensor(positions, dtype=torch.int64, device=self.device)
        return input_ids, positions, context

        
    def prepare_decode(self, seqs: list[Sequence]):
        """
        从每个 seq 里取 last_token
        组成 input_ids
        计算每个 seq 当前最后 token 的 positions
        如果是单卡简易版，最小输出通常就是：

        input_ids: shape [batch]
        positions: shape [batch]

        """
        input_ids = []
        positions = []
        context_lens = []
        slot_mapping = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))  # 每个seq的上下文长度

            block_id = seq.block_table[-1]
            offset = seq.last_block_num_tokens - 1
            slot_mapping.append(block_id * seq.block_size + offset)
        
        block_tables = self.prepare_block_tables(seqs)
        context = get_context(is_prefill=False, 
            cu_seqlens_q=None, 
            cu_seqlens_k=None, 
            max_seqlen_q=0, 
            max_seqlen_k=0,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, device=self.device), 
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device=self.device),
            block_tables=block_tables
            )
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        positions = torch.tensor(positions, dtype=torch.int64, device=self.device)
        return input_ids, positions, context

    def run(self, seqs, is_prefill:bool):
        if is_prefill:
            # input_ids:[token_len]
          input_ids, position, context = self.prepare_prefill(seqs)
        else:
          # input_ids:[batch]
          input_ids, position, context = self.prepare_decode(seqs)
        logits = self.model(input_ids, position, context)
        temperatures = self.prepare_sampler(seqs)
        token_ids = self.sampler(logits, temperatures)
        return token_ids
        
