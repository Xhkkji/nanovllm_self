

class Sequence:
    """
    序列类，代表一个推理请求
    一个Sequence实例对应一个batch，多batch对应多个sequence实例
    """
    def __init__(self, seq_idx, token_ids):
        self.seq_idx = seq_idx  # 序列id，整个token的ids
        self.token_ids = token_ids
        self.block_table = []  # 物理块ID列表
        self.num_cached_tokens = 0  # 初始为 0， 记录有多少 token 已经存在于 KV Cache 中（通过前缀共享获得），不需要重复计算。
        self.num_prompt_tokens = len(token_ids)  # prompt 长度
        self.finished = False  # 是否完成

    def append_token(self, token):
        """追加新生成的 token"""
        self.token_ids.append(token)