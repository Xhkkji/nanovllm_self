import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from nanovllm.config import Config
from ..models.qwen3 import Qwen3Model
from transformers import AutoConfig
from nanovllm.engine.block_manager import block_manager
from nanovllm.engine.Sequence import Sequence

class ModelRunner(nn.Module):
    def __init__(self, config:Config, block_manager):
        super().__init__()
        self.config = config
        self.device = self.config.device
        model_config = AutoConfig.from_pretrained("/home/xhk/model/Qwen3-0.6B/")
        self.model = Qwen3Model(model_config).to(self.device)
        self.block_manager = block_manager


    def prepare_prefill(self, seqs: list[Sequence]) -> (list[Sequence], list[list[int]]):
        """
        seqA.token_ids = [11, 12, 13, 14, 15]
        seqB.token_ids = [21, 22, 23]
        打包成
        input_ids = [11, 12, 13, 14, 15, 21, 22, 23]
        positions = [0, 1, 2, 3, 4, 0, 1, 2]
        
        目前先实现简化版,以seq为单位，先不铺平
        """
        # seqs_len = len(seqs)
        # input_seq = []
        # position = []

        # for seq in seqs:
        #     position.extend([i for i in range(len(seq))])
        #     input_seq.extend(seq)
        # return input_seq, position
        
        seqs_len = len(seqs)
        input_seq = []
        position = []

        for seq in seqs:
            position.append([i for i in range(len(seq))])
            input_seq.append(seq)
        return input_seq, position
    
    def prepare_decode(self, seqs: list[Sequence]) -> (list[int], list[int]):
        """
        从每个 seq 里取 last_token
        组成 input_ids
        计算每个 seq 当前最后 token 的 positions
        如果是单卡简易版，最小输出通常就是：

        input_ids: shape [batch]
        positions: shape [batch]
        批量逻辑暂未实现，因此直接返回list[seq]和position[]
        """
        # input_ids = []
        # position = []
        # for seq in seqs:
        #     input_ids.append(seq.last_token)
        #     position.append(len(seq)-1)
        # return input_ids, position
        
        input_ids = []
        position = []
        for seq in seqs:
            input_ids.append(seq)
            position.append(len(seq)-1)
        return input_ids, position
    
    def run(self, li, position, is_prefill: bool):
        # 处理一批seq
        last_logits = []
        if is_prefill:
            seq_list = li  # 一维列表，所有seq的token拼成一列同时处理
            for i, seq in enumerate(seq_list):
                # torch.Size([token_ids.len, 151936])
                outputs = self.model(torch.tensor(seq.token_ids, device=self.config.device), positions=torch.tensor(position[i], device=self.config.device), block_manager=self.block_manager, seq=seq, is_prefill=is_prefill)
                last_logits.append(outputs[-1])
            # print(f'last_logits{last_logits}')
            return last_logits  # [seq_num, vocab_dim]
        else:
            # 批量暂未实现，先逐seq处理
            seq_list = li
            for i, seq in enumerate(seq_list):
                outputs = self.model(torch.tensor([seq.last_token], device=self.config.device), positions=torch.tensor(position[i], device=self.config.device), block_manager=self.block_manager, seq=seq, is_prefill=is_prefill)
                # print(f'outputs:{outputs}')
                # outputs:[1, vocab_size]
                last_logits.append(outputs)
            # print(f'last_logits{last_logits}')
            return last_logits  # [seq_num, vocab_dim]
        


    # def prepare_block_tables(self, seq_list: list[int]):
    #     print("\n创建 BlockManager...")
    #     self.block_manager = block_manager(
    #         num_blocks=self.config.num_blocks,
    #         block_size=self.config.block_size,
    #         num_layers=self.model.num_layers,
    #         num_kv_heads=self.model.num_kv_heads,
    #         head_dim=self.model.head_dim
    #     )
    #     print("✅ BlockManager 创建成功")
    
    def sample(self, logits, seqs) -> list[int]:
        """
        它的输入不是 Sequence 状态本身，而是：

        本轮每条 seq 对应的最终 logits
        以及这些 seq 的采样参数(比如temperature)
        它的输出应该是：

        list[int]
        长度和 seqs 一样
        每个 int 对应一条 seq 的 next token
        """
        seq_next_tokens = []
        for i, seq in enumerate(seqs):
            temperature = seq.temperature
            outputs_logits = logits[i]
            if temperature > 0:
                outputs_logits = outputs_logits / temperature
                probs = torch.softmax(outputs_logits, dim=-1)
            # next_token = torch.argmax(outputs.logits[0, -1, :])
            # 按照概率分布随机取1个，next_tokens为所有batch的下一个token
                next_tokens = torch.multinomial(probs, num_samples=1).item()  # [1]
            else:
                next_tokens = torch.argmax(outputs_logits).item()  # [1]
            seq_next_tokens.append(next_tokens)
        return seq_next_tokens
        
