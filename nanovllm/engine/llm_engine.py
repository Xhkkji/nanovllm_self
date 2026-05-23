from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from nanovllm.config import Config
from ..models.qwen3 import Qwen3Model
from transformers import AutoConfig
from nanovllm.engine.block_manager import block_manager as bm
from nanovllm.engine.Sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

config = Config(
    model_path="/home/xhk/model/Qwen3-0.6B/",
    max_num_seqs=256,              # 同时处理 256 个请求
    max_num_batched_tokens=16384,  # 每批最多 16384 tokens
    max_model_len=4096,            # 每个请求最长 4096 tokens
    gpu_memory_utilization=0.9,    # 使用 90% 显存
    block_size=16,         # PagedAttention 块大小
    device="cuda:0"
)

class llm_engine():
  def __init__(self, tensor_parallel_size=1):
    # self.text = text
    self.tokenizer = AutoTokenizer.from_pretrained('/home/xhk/model/Qwen3-0.6B')
    self.model = AutoModelForCausalLM.from_pretrained("/home/xhk/model/Qwen3-0.6B",
      torch_dtype=torch.bfloat16,
      device_map=config.device)
    self.config = config
    self.tensor_parallel_size = tensor_parallel_size
    self.eos_token_id = self.tokenizer.eos_token_id
  def encoder(self, text):
    return self.tokenizer(text)

  def generate(self, input, max_token = 20, temperature=0.8):
    batch_size = input.shape[0]

    # 跟踪哪些batch已经结束
    finished = torch.zeros(batch_size, dtype=torch.bool, device=self.config.device)

    past_key_values = None
    all_tokens = input.to(self.config.device)
    for i in tqdm(range(max_token)):
      if finished.all():
        break

      with torch.no_grad():
        if past_key_values is None:
            current_tokens = all_tokens
        else:
            current_tokens = all_tokens[:, -1:]  # 取最后一个token
        outputs = self.model(current_tokens, past_key_values=past_key_values, use_cache=self.config.use_cache)
        # outputs:(batchsize, token_num, vocab_len)

        # outputs.logits.shape:batchsize, seq_len, vocab_size
        # 对于logits，其输出为batch、all_tokens长度+1，即每个单词都会预测下一个单词，但是最后一个才是all_token的下一个单词，从最后一维(vocab_len)中选出概率最大的词元
        # 引入温度采样
        outputs_logits = outputs.logits[:, -1, :]  # 取最后一个词元[batch, 1, vocab_size]
        if self.config.use_cache:
            past_key_values = outputs.past_key_values
        # print(f'past_key_values:{past_key_values}')
        if temperature > 0:
          outputs_logits /= temperature
          probs = torch.softmax(outputs_logits, dim=-1)
          # next_token = torch.argmax(outputs.logits[0, -1, :])
          # 按照概率分布随机取1个，next_tokens为所有batch的下一个token
          next_tokens = torch.multinomial(probs, num_samples=1)  # [batch, 1]
        else:
          next_tokens = torch.argmax(outputs_logits, dim=-1, keepdim=True)  # [batch, 1]

        newly_finished = (next_tokens.squeeze(-1) == self.eos_token_id)  # 查看哪个batch序列已经生成结束符
        finished = finished | newly_finished
        # 将finished[false, true, false]加第一维，重构为[[false],[true],[false]],与next_tokens对齐
        next_tokens = torch.where(finished.unsqueeze(1),  # condition: 条件判断
                    torch.tensor(self.eos_token_id, device=self.config.device).repeat(batch_size, 1),  # x: 条件为True时取这个值
                    next_tokens)  # y: 条件为False时取这个值

        all_tokens = torch.cat([all_tokens, next_tokens], dim=1)  # [batch, seq_len+1]
    return all_tokens

  def decode(self, all_tokens):
    # return self.tokenizer.decode(all_tokens)
    return self.tokenizer.decode(all_tokens[0], skip_special_tokens=True)

class llm_engine_self():
  def __init__(self, tensor_parallel_size=1):
    # self.text = text
    print("llm_engine_self..")
    self.tokenizer = AutoTokenizer.from_pretrained('/home/xhk/model/Qwen3-0.6B')
    model_config = AutoConfig.from_pretrained("/home/xhk/model/Qwen3-0.6B/")
    self.model = Qwen3Model(model_config).to(config.device)
    self.config = config
    self.tensor_parallel_size = tensor_parallel_size
    self.eos_token_id = self.tokenizer.eos_token_id

    """
    generate处理的是一整个流程，bm和seq应该在这里创建
    """
    print("\n创建 BlockManager...")
    self.block_manager = bm(
        num_blocks=self.config.num_blocks,
        block_size=self.config.block_size,
        num_layers=self.model.num_layers,
        num_kv_heads=self.model.num_kv_heads,
        head_dim=self.model.head_dim
    )
    print("✅ BlockManager 创建成功")
    self.scheduler = Scheduler(config, self.block_manager)
    self.model_runner = ModelRunner(config, self.block_manager)

  def encoder(self, text):
    return self.tokenizer(text)
  
  # def step(self):
    

  def generate(self, input):
    seq = None
    finished_seqs = []
    try:
      print("\n创建Sequence...")
      print(f"  Prompt tokens: {input}")
      seq = Sequence(seq_idx=0, token_ids=input[0].tolist())  # 取出 token 列表,没有batch维度
      # seq.block_size = self.block_manager.block_size  # 添加 block_size 属性
      # seq.block_table = self.block_manager.allocate_with_prefill(seq)  # 分配块并进行前缀共享, 将分配的块表关联到序列
      # print(f"✅ Sequence 创建成功，分配块ID: {seq.block_table}")

      self.scheduler.add(seq)
      while not self.scheduler.is_finished():
        seq_list, is_prefill = self.scheduler.schedule()
        if is_prefill:
          input_list, position = self.model_runner.prepare_prefill(seq_list)
        else:
          # 批量逻辑还没实现
          input_list, position = self.model_runner.prepare_decode(seq_list)

        outputs_logits = self.model_runner.run(input_list, position, is_prefill)
        seq_next_tokens = self.model_runner.sample(outputs_logits, seq_list)
        
        # 将新生成的token加入seq，并根据block是否已满更新block，下一轮训练会将新token的kv存入kvcache
        finished_seqs.extend(self.scheduler.postprocess(seq_list, seq_next_tokens)) # 更新 seq 的 token 列表，供 block_manager 存储 KV 时使用

    finally:
      if seq is not None and getattr(seq, "block_table", None):
        self.block_manager.free_blocks(seq.block_table)
        output = []
        for seq in finished_seqs:
          output.append(seq.token_ids)
        return output
      


    # seq = None
    # try:
    #   print("\n创建Sequence...")
    #   print(f"  Prompt tokens: {input}")
    #   seq = Sequence(seq_idx=0, token_ids=input[0].tolist())  # 取出 token 列表
    #   seq.block_size = self.block_manager.block_size  # 添加 block_size 属性
    #   seq.block_table = self.block_manager.allocate_with_prefill(seq)  # 分配块并进行前缀共享, 将分配的块表关联到序列
    #   print(f"✅ Sequence 创建成功，分配块ID: {seq.block_table}")

    #   batch_size = input.shape[0]
    #   # 跟踪哪些batch已经结束
    #   finished = torch.zeros(batch_size, dtype=torch.bool, device=self.config.device)
    #   # all_tokens:[batch, seq_len]
    #   all_tokens = input.to(self.config.device)
    #   is_prefill = True

    #   for i in tqdm(range(max_token)):
    #     if finished.all():
    #       break

    #     with torch.no_grad():
    #       if is_prefill:
    #         current_tokens = all_tokens[0]  # prefill阶段输入整个序列，current_tokens为[seq_len]
    #         positions = torch.arange(0, len(seq.token_ids), device=self.config.device).unsqueeze(0)
    #         outputs = self.model(current_tokens, positions=positions, block_manager=self.block_manager, seq=seq, is_prefill=True)
    #         outputs_logits = outputs[-1, :].unsqueeze(0)  # [batch, token_len, vocab_dim]
    #         is_prefill = False
    #       else:
    #         current_tokens = all_tokens[0, -1:]  # 取最后一个token
    #         # outputs:(token_num, vocab_len)
    #         outputs = self.model(current_tokens, positions=torch.tensor([[len(seq.token_ids)-1]], device=self.config.device), block_manager=self.block_manager, seq=seq, is_prefill=False)
    #         outputs_logits = outputs.unsqueeze(0)  # [batch, token_len, vocab_dim]

    #       # outputs.logits.shape:batchsize, seq_len, vocab_size
    #       # 对于logits，其输出为batch、all_tokens长度+1，即每个单词都会预测下一个单词，但是最后一个才是all_token的下一个单词，从最后一维(vocab_len)中选出概率最大的词元
    #       # 引入温度采样
    #       # print(f"outputs_logits shape: {outputs_logits.shape}")  # 添加调试输出，查看 logits 的形状
    #       # outputs_logits = outputs[:, -1, :]  # 取最后一个词元[batch, 1, vocab_size]
    #       if temperature > 0:
    #         outputs_logits /= temperature
    #         probs = torch.softmax(outputs_logits, dim=-1)
    #       # next_token = torch.argmax(outputs.logits[0, -1, :])
    #       # 按照概率分布随机取1个，next_tokens为所有batch的下一个token
    #         next_tokens = torch.multinomial(probs, num_samples=1)  # [batch, 1]
    #       else:
    #         next_tokens = torch.argmax(outputs_logits, dim=-1, keepdim=True)  # [batch, 1]

    #       newly_finished = (next_tokens.squeeze(-1) == self.eos_token_id)  # 查看哪个batch序列已经生成结束符
    #       finished = finished | newly_finished
    #       # 将finished[false, true, false]加第一维，重构为[[false],[true],[false]],与next_tokens对齐
    #       next_tokens = torch.where(finished.unsqueeze(1),  # condition: 条件判断
    #                   torch.tensor(self.eos_token_id, device=self.config.device).repeat(batch_size, 1),  # x: 条件为True时取这个值
    #                   next_tokens)  # y: 条件为False时取这个值
    #       # print(f"next_tokens: {next_tokens}")  # 添加调试输出，查看 next_tokens 的值
    #       all_tokens = torch.cat([all_tokens, next_tokens], dim=1)  # [batch, seq_len+1]
          
    #       # 将新生成的token加入seq，并根据block是否已满更新block，下一轮训练会将新token的kv存入kvcache
    #       new_token_id = next_tokens[0, 0].item()
    #       seq.append_token(new_token_id)  # 更新 seq 的 token 列表，供 block_manager 存储 KV 时使用
    #       if len(seq.token_ids) > len(seq.block_table) * self.block_manager.block_size:
    #           # print(f"Warning: seq长度超过已分配块的容量，可能需要分配更多块")
    #           new_block_id = self.block_manager.allocate_block(1)[0]  # 分配一个新块
    #           seq.block_table.append(new_block_id)  # 更新 seq 的块表

    #   return all_tokens
      
    # finally:
    #   if seq is not None and getattr(seq, "block_table", None):
    #     self.block_manager.free_blocks(seq.block_table)
      
      

  def decode(self, all_tokens):
    # return self.tokenizer.decode(all_tokens)
    print(f'all_token:{all_token}')
    return self.tokenizer.decode(all_tokens[0], skip_special_tokens=True)
