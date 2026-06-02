from enum import Enum, auto
import torch
from nanovllm.sampling_params import SamplingParams

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    """
    序列类，代表一个推理请求
    一个Sequence实例对应一个batch，多batch对应多个sequence实例
    """
    def __init__(self, seq_idx: int, token_ids: list[int], sampling_params = SamplingParams(), block_size=16):
        self.seq_idx = seq_idx  # 序列id，整个token的ids
        self.token_ids = token_ids
        self.token_len = len(token_ids)
        self.last_token = token_ids[-1]
        self.block_table = []  # 物理块ID列表
        self.block_size = block_size
        # num_prompt_tokens 看作一个序列的初始尺寸，而 num_cached_tokens 是一个进度指针，记录已经往前推进了多少
        self.num_prompt_tokens = len(token_ids)  # prompt 长度,全程不变，用于计算已经生成token的数量
        self.num_cached_tokens = 0  # 初始为 0， 记录有多少 token 已经存在于 KV Cache 中（通过前缀共享获得），不需要重复计算。
        self.finished = False  # 是否完成
        self.status = SequenceStatus.WAITING
        
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens  # 当前seq最多生成多少个token
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.token_len

    def __getitem__(self, idx):
        return self.token_ids[idx]

    def append_token(self, token):
        """追加新生成的 token"""
        # print(type(token), token)
        self.token_ids.append(token)
        self.token_len += 1
        self.last_token = self.token_ids[-1]

    @property
    def num_blocks(self):
        return (len(self.token_ids) + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        rem = len(self.token_ids) % self.block_size
        if rem == 0:
            return self.block_size
        else:
            return rem

    @property
    def num_completion_tokens(self):
        return len(self.token_ids) - self.num_prompt_tokens
