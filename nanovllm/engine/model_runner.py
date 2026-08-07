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
from contextlib import contextmanager

class ModelRunner(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.device = self.config.device
        model_config = AutoConfig.from_pretrained("/home/xhk/model/Qwen3-0.6B/")
        self.model = Qwen3Model(model_config).to(self.device)
        self.sampler = Sampler()
        # print("\n创建 BlockManager...")
        self.block_manager = block_manager(
            num_blocks=self.config.num_blocks,
            block_size=self.config.block_size,
            num_layers=self.model.num_layers,
            num_kv_heads=self.model.num_kv_heads,
            head_dim=self.model.head_dim
        )

        # 判断精度是否支持
        self._validate_kv_cache_dtype()
        # print("✅ BlockManager 创建成功")
        # 分配kv_cache
        # 形状: [2(key和value), num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_cache = torch.zeros(
            2,
            self.model.num_layers,
            self.config.num_blocks,
            self.config.block_size,
            self.model.num_kv_heads,
            self.model.head_dim,
            dtype=config.kv_cache_dtype,
            device=config.device
        )
        
        self.kv_scale_cache = None
        if self.config.kv_cache_quant_mode == "int8_mock":
            # per-token-per-kv-head scale
            # shape: [2, num_layers, num_blocks, block_size, num_kv_heads, 1]
            #
            # 2 表示 K / V 两份 scale：
            #   0 -> K scale
            #   1 -> V scale
            #
            # 最后一维是 1，表示每个 token、每个 kv head 共享一个 scale，
            # 这个 scale 会 broadcast 到 head_dim。
            self.kv_scale_cache = torch.ones(
                2,
                self.model.num_layers,
                self.config.num_blocks,
                self.config.block_size,
                self.model.num_kv_heads,
                1,
                dtype=self.config.kv_cache_scale_dtype,
                device=self.config.device,
            )
        
        self.bind_kvcache_to_attention()

        # 计算图
        # self.graph_bucket = [1, 2, 4, 8]
        max_bs = self.config.max_num_seqs
        self.graph_bucket = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graph_states = {}
        self.enable_cuda_graph = False   # 开关
        if self.enable_cuda_graph:
            self.init_graph_states()

        # self.decode_graph = None        # 存储捕获的 CUDA Graph 对象
        # self.graph_batch_size = 1       # 固定 batch_size=1（第一版简化）
        # self.graph_context = None       # 存储 context 信息
        # self.graph_input_ids = None     # 存储输入 token ID
        # self.graph_positions = None     # 存储位置信息
        # self.graph_logits = None        # 存储输出结果

    def init_graph_states(self):
        """
        针对bucket
        为每一个预设的 Batch Size，都预先分配好一套“占位张量”（即 Graph State），
        以便后续可以随时对这些状态进行 CUDA Graph 的捕获（Capture）
        """
        for bs in self.graph_bucket:
            self.graph_states[bs] = self._allocate_graph_state(bs)

    def _allocate_graph_state(self, batch_size: int):
        """
        根据bz生成捕获图时需要的信息

        第一版：
        - 只 capture decode
        - 只支持固定 batch_size
        - 只支持当前稳定主链：torch prefill + flash decode
        """
        # self.config.max_model_len: 模型能够处理的最大 token 数量
        max_num_blocks = (self.config.max_model_len + self.config.block_size - 1) // self.config.block_size

        graph_input_ids = torch.zeros(
            batch_size,
            dtype=torch.int64,
            device=self.device,
        )

        graph_seq_need_compute_logits = torch.arange(
            batch_size,
            dtype=torch.int32,
            device=self.device,
        )

        graph_cu_seqlens_q = torch.arange(
            0,
            batch_size+1,
            dtype=torch.int32,
            device=self.device,
        )

        graph_cu_seqlens_k = torch.zeros(
            batch_size + 1,
            dtype=torch.int32,
            device=self.device
        )

        graph_positions = torch.zeros(
            batch_size,
            dtype=torch.int64,
            device=self.device,
        )

        graph_slot_mapping = torch.zeros(
            batch_size,
            dtype=torch.int32,
            device=self.device,
        )

        graph_context_lens = torch.zeros(
            batch_size,
            dtype=torch.int32,
            device=self.device,
        )

        graph_block_tables = torch.full(
            (batch_size, max_num_blocks),
            fill_value=-1,
            dtype=torch.int32,
            device=self.device,
        )

        graph_context = get_context(
            cu_seqlens_q=graph_cu_seqlens_q,
            cu_seqlens_k=graph_cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=1,
            slot_mapping=graph_slot_mapping,
            context_lens=graph_context_lens,
            block_tables=graph_block_tables,
            seq_need_compute_logits=graph_seq_need_compute_logits,
        )

        return {
            "captured": False,
            "graph": None,
            "input_ids": graph_input_ids,
            "positions": graph_positions,
            "cu_seqlens_q": graph_cu_seqlens_q,
            "cu_seqlens_k": graph_cu_seqlens_k,
            "slot_mapping": graph_slot_mapping,
            "context_lens": graph_context_lens,
            "block_tables": graph_block_tables,
            "seq_need_compute_logits": graph_seq_need_compute_logits,
            "context": graph_context,
            "logits": None,
        }

    def _set_attention_capture_mode(self, enabled: bool):
        """
        设置是否使用capture_map
        """
        for layer in self.model.layers:
            layer.p_attn.in_cuda_graph_capture = enabled

    @contextmanager
    def _cuda_graph_capture_guard(self):
        self._set_attention_capture_mode(True)
        try:
            # 此处交给with代码块执行
            yield
        finally:
            # with代码块执行完毕之后执行
            self._set_attention_capture_mode(False)

    def select_graph_bucket(self, batch_size: int):
        """
        向上取整选择 bucket。
        例如:
        1 -> 1
        2 -> 2
        3 -> 4
        4 -> 4
        5 -> 8
        8 -> 8

        如果没有可用 bucket，比如 batch_size=9，而最大 bucket=8，
        则返回 None，调用方走 eager fallback。
        """
        for bs in self.graph_bucket:
            if batch_size <= bs:
                return bs
        return None

    def copy_decode_graph_to_bucket(self, state, input_ids, positions, context, real_bs: int):
        """
        把真实 decode 输入拷到固定 bucket 的静态张量里。

        state:
            某个 bucket 对应的 graph state
        real_bs:
            当前真实 batch size
        """
        # 先整体清成“合法占位数据”
        # 这样 bucket 中多出来的那些 dummy 行也能安全参与 replay
        state["input_ids"].zero_()
        state["positions"].zero_()
        state["slot_mapping"].zero_()
        state["context_lens"].fill_(1)
        state["block_tables"].fill_(-1)

        state["input_ids"][:input_ids.numel()].copy_(input_ids)
        state["positions"][:positions.numel()].copy_(positions)
        state["slot_mapping"][:context.slot_mapping.numel()].copy_(context.slot_mapping)
        state["context_lens"][:real_bs].copy_(context.context_lens)

        # q_len 如果你当前只 capture decode steady-state，可以固定每条 seq 都是 1
        state["cu_seqlens_q"].copy_(
            torch.arange(0, real_bs + 1, dtype=torch.int32, device=self.device)
        )

        # k_len 用 context_lens 做前缀和
        state["cu_seqlens_k"][0] = 0
        state["cu_seqlens_k"][1:real_bs + 1] = torch.cumsum(context.context_lens, dim=0)

        state["block_tables"][:real_bs, :context.block_tables.size(1)].copy_(context.block_tables)


    def capture_decode_graph(self, batch_size: int):
        """
        for bucket
        """
        state = self.graph_states[batch_size]  # 从已经分配好的图状态中取出对应batchsize

        if state["captured"]:
            return

        input_ids = state["input_ids"]
        positions = state["positions"]
        context = state["context"]

        # 给一份“合法但无意义”的 warmup 数据
        # 这里只是为了让 kernel / lazy init 提前完成
        input_ids.zero_()
        positions.zero_()
        state["slot_mapping"].zero_()
        state["context_lens"].fill_(1)
        state["block_tables"].fill_(0)

        print(f"graph warmup for bs={batch_size}..")
        for _ in tqdm(range(3)):
            _ = self.model(input_ids, positions, context)
        torch.cuda.synchronize()

        # # torch.cuda.CUDAGraph() 创建一个空白的"录制器"。
        # # with torch.cuda.graph(self.decode_graph): 开启录制模式。
        # # 在这个上下文中的所有 CUDA 操作，都不会真正执行，而是被"录"进 graph 中。
        # # 包括：矩阵乘法、注意力计算、KVCache 写入、LayerNorm、激活函数... 整个模型前向传播。
        # # self.model(...) 被调用，但不会真正计算，而是把计算流程录制下来。
        # # 录制完成后，self.graph_logits 中存储的是捕获的输出张量（也是假数据，因为输入是假的）。
        graph = torch.cuda.CUDAGraph()

        with self._cuda_graph_capture_guard():
            with torch.cuda.graph(graph):
                state["logits"] = self.model(input_ids, positions, context)

        state["captured"] = True
        state["graph"] = graph


    def bind_kvcache_to_attention(self):
        """
        把全局 kv_cache 中每一层对应的视图，提前绑定到该层的 PagedAttention 上。
        这样前向时就不用每次手动传全局 cache。
        把全局 kv_cache 中每一层对应的视图绑定到 PagedAttention。
        kv_cache_dtype 以实际 cache dtype 为准，避免 config/cache 不一致。
        """
        for layer_id, layer in enumerate(self.model.layers):
            # 取出对应qwendecodelayer中的p_attn
            p_attn = layer.p_attn

            p_attn.k_cache = self.kv_cache[0, layer_id]
            p_attn.v_cache = self.kv_cache[1, layer_id]
            p_attn.block_size = self.config.block_size
            p_attn.layer_id = layer_id

            # 新增
            p_attn.kv_cache_quant_mode = self.config.kv_cache_quant_mode
            p_attn.kv_cache_dtype = self.config.kv_cache_dtype
            p_attn.attention_compute_dtype = self.config.attention_compute_dtype
            # p_attn.decode_backend = self.config.decode_attention_backend
            if self.config.kv_cache_quant_mode == "int8_mock":
                p_attn.forward_backend = "torch"
            
            if self.kv_scale_cache is not None:
                p_attn.k_scale_cache = self.kv_scale_cache[0, layer_id]
                p_attn.v_scale_cache = self.kv_scale_cache[1, layer_id]
            else:
                p_attn.k_scale_cache = None
                p_attn.v_scale_cache = None


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

    def prepare_sampler(self, seqs:list[Sequence], context):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).to(self.config.device)
        if context.seq_need_compute_logits.numel() > 0:
            temperatures = temperatures[context.seq_need_compute_logits]
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

    def prepare_model_input(self, seqs: list[Sequence]):
        """
        合并prefill和decode的数据准备过程
        seqA.token_ids = [11, 12, 13, 14, 15]
        seqB.token_ids = [21, 22, 23]
        打包成
        input_ids = [11, 12, 13, 14, 15, 21, 22, 23]
        positions = [0, 1, 2, 3, 4, 0, 1, 2]
        """
        input_ids = []
        positions = []
        # 构建cu_seqlen
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        context_lens = []  # 所有seq的上下文长度，[[seq1], [seq2]....]
        seq_need_compute_logits = []
        block_tables = None

        for seq_index, seq in enumerate(seqs):
            start = seq.num_cached_tokens
            end = seq.num_cached_tokens + seq.num_new_tokens  # 不能直接取长度？？
            seqlen_q = seq.num_new_tokens  # 本轮 Query 的长度，即本轮新生成的 token 数量
            seqlen_k = seq.num_context_tokens  # 本轮 Key 的长度，即当前序列已有的全部 token 数量

            # 本轮可见的上下文长度
            # 关键：无论当前这条 seq 是 chunked prefill 还是 decode，
            # 都统一记录“本轮完整可见 KV 长度”
            # get_kv_cache里面可以直接利用context_len赋值
            context_lens.append(seqlen_k)

            if seq.is_need_logits:
                # 对应即将生成新token的seq场景
                seq_need_compute_logits.append(seq_index)

            input_ids.extend(seq.token_ids[start:end])
            positions.extend(range(start, end))

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

            for token_pos in range(start, end):
                block_idx = token_pos // seq.block_size
                offset = token_pos % seq.block_size
                block_id = seq.block_table[block_idx]
                # 按顺序对应block索引
                slot_mapping.append(block_id * seq.block_size + offset)

        # 当 seqlen_k > seqlen_q 时，意味着当前序列在计算注意力时
        # 需要复用（即查询）之前已经计算并缓存的 KV Cache
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            # 生成带填充的seqs的二维表
            block_tables = self.prepare_block_tables(seqs)

        context = get_context(
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, device=self.device),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device=self.device),
            block_tables=block_tables,
            seq_need_compute_logits=torch.tensor(seq_need_compute_logits, dtype=torch.int32, device=self.device),
        )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        positions = torch.tensor(positions, dtype=torch.int64, device=self.device)
        return input_ids, positions, context

    def run(self, seqs: list[Sequence]):
        # input_ids:[token_len / batch]
        input_ids, positions, context = self.prepare_model_input(seqs)
        logits = self.model(input_ids, positions, context)

        # seq_need_compute_logits存的是需要logit操作的seq_index
        # 也就是记录哪些seq是需要生成下一个token的
        if context.seq_need_compute_logits.numel() > 0:
            # qwen的前向过程已经实现：只取需要logits的部分token
            # sampled_logits = logits[context.seq_need_compute_logits]
            temperatures = self.prepare_sampler(seqs, context)
            token_ids = self.sampler(logits, temperatures)
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.reshape(-1).tolist()
            token_ids = [int(x) for x in token_ids]
        else:
            token_ids = []

        seq_need_compute_logits = context.seq_need_compute_logits.tolist()
        return token_ids, seq_need_compute_logits

    def _validate_kv_cache_dtype(self):
        supported = {
            torch.float32,
            torch.float16,
            torch.bfloat16,
        }
        if self.config.kv_cache_quant_mode == "int8_mock":
            supported.add(torch.int8)

        if self.config.kv_cache_dtype not in supported:

            raise ValueError(
                f"unsupported kv_cache_dtype: {self.config.kv_cache_dtype}"
            )

        # flash-attn paged cache 通常只支持 fp16 / bf16。
        # fp32 KV cache 第一版建议走 torch backend 做正确性测试。
        if (
            self.config.kv_cache_dtype == torch.float32
            and getattr(self.config, "decode_backend", "flashattn") == "flashattn"
        ):
            print(
                "[WARN] fp32 kv_cache_dtype may not be supported by flash-attn; "
                "use torch attention path for fp32 correctness test."
            )

