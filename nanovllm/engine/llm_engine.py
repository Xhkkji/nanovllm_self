from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from time import perf_counter
from nanovllm.config import Config
from ..models.qwen3 import Qwen3Model
from transformers import AutoConfig
from nanovllm.engine.block_manager import block_manager as bm
from nanovllm.engine.Sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.sampling_params import SamplingParams
# TP
from nanovllm.distributed.parallel_state import init_tensor_parallel
import os

config = Config(
    model_path="/home/xhk/model/Qwen3-0.6B/",
    max_num_seqs=256,              # 同时处理 256 个请求
    max_num_batched_tokens=16384,  # 每批最多 16384 tokens
    max_model_len=4096,            # 每个请求最长 4096 tokens
    gpu_memory_utilization=0.9,    # 使用 90% 显存
    block_size=256,         # PagedAttention 块大小
    device="cuda:0"
)

class llm_engine():
  def __init__(self, tensor_parallel_size=1):
    # # 设置多卡推理环境
    # init_tensor_parallel(tensor_parallel_size)
    # if tensor_parallel_size > 1:
    #   local_rank = int(os.environ["LOCAL_RANK"])
    #   self.config.device = f"cuda:{local_rank}"
    # else:
    #   self.config.device = "cuda:0"
    
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
  def __init__(self, model_path="/home/xhk/model/Qwen3-0.6B", tensor_parallel_size=1, enable_profile=False):
    # print("llm_engine_self..")
    # TP 最小主线：LLM_self 这条自研 runtime 主链也必须先初始化
    # torch.distributed，并把当前进程绑定到 LOCAL_RANK 对应的 GPU。
    init_tensor_parallel(tensor_parallel_size)

    self.config = config
    self.config.model_path = model_path
    self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
    
    if tensor_parallel_size > 1:
      local_rank = int(os.environ["LOCAL_RANK"])
      self.config.device = f"cuda:{local_rank}"
    else:
      self.config.device = "cuda:0"
      
    self.tensor_parallel_size = tensor_parallel_size
    self.enable_profile = enable_profile
    self.eos_token_id = self.tokenizer.eos_token_id

    self.model_runner = ModelRunner(config)
    self.scheduler = Scheduler(config, self.model_runner.block_manager)
  
  def _set_attention_profile(self, enabled: bool):
    """
    把profile开关传入p_attn
    """
    for layer in self.model_runner.model.layers:
        layer.p_attn.enable_profile = enabled

  def encoder(self, text):
    return self.tokenizer(text)
  
  def generate(self, inputs, sampling_params=None, return_metrics=False, request_metadata: dict | None = None):
    if sampling_params is None:
      sampling_params = [SamplingParams()] * len(inputs)
    
    # Agent-aware prompt caching：
    # Agent / Backend 会通过 request_metadata 传入 session_id。
    # 同一个 session_id 的多轮请求可以在 block_manager 里查找并复用前缀 KV blocks。
    session_id = None
    if request_metadata is not None:
      session_id = request_metadata.get("session_id") or request_metadata.get("program_id")

    all_seqs = []
    # print("\n创建Sequence...")
    # print(f"  Prompt tokens: {inputs}")
    for i, input in enumerate(inputs):
      seq = Sequence(seq_idx=i, token_ids=input.tolist(), sampling_params=sampling_params[i], block_size=self.config.block_size, session_id=session_id)  # 取出 token 列表,没有batch维度
      self.scheduler.add(seq)
      all_seqs.append(seq)
    
    # 设置attention中是否要输出profile
    self._set_attention_profile(self.enable_profile)

    # pbar = tqdm(total=seq.max_tokens, desc="Generating", dynamic_ncols=True)
    # 多 seq 时，进度条总量用所有 seq 的 max_tokens 总和更合理
    total_max_tokens = sum(seq.max_tokens for seq in all_seqs)
    total_prompt_tokens = sum(seq.num_prompt_tokens for seq in all_seqs)
    pbar = tqdm(total=total_max_tokens, desc="Generating", dynamic_ncols=True)
    total_time = 0.0
    schedule_steps = 0
    measure_timing = self.enable_profile or return_metrics

    with torch.inference_mode():
      while not self.scheduler.is_finished():
        # 统计本轮调度前，所有 seq 一共已经生成了多少 completion token
        prev_completion = sum(seq.num_completion_tokens for seq in all_seqs)

        seqs = self.scheduler.schedule()  # 此处已经对seq的token分配了block
        if not seqs:
          continue
        
        if measure_timing:
          torch.cuda.synchronize()
          t0 = perf_counter()
        token_ids, seq_need_compute_logits = self.model_runner.run(seqs)
        if measure_timing:
          torch.cuda.synchronize()
          total_time += perf_counter() - t0
          schedule_steps += 1
        # 规范化处理，保证加入seq的是列表而不是tensor
        if isinstance(token_ids, torch.Tensor):
          token_ids = token_ids.reshape(-1).tolist()
        token_ids = [int(x) for x in token_ids]
        # print(token_ids, [type(x) for x in token_ids])

        # 将新生成的token加入seq，并根据block是否已满更新block，下一轮训练会将新token的kv存入kvcache
        self.scheduler.postprocess(seqs, token_ids, seq_need_compute_logits)# 更新 seq 的 token 列表，供 block_manager 存储 KV 时使用
        new_completion = sum(seq.num_completion_tokens for seq in all_seqs)
        
        pbar.update(new_completion - prev_completion)

    pbar.close()
    if self.enable_profile:
      avg_schedule_step = total_time / schedule_steps if schedule_steps > 0 else 0.0
      total_store = sum(layer.p_attn.profile_decode["store"] for layer in self.model_runner.model.layers)
      total_load = sum(layer.p_attn.profile_decode["load"] for layer in self.model_runner.model.layers)
      total_attn = sum(layer.p_attn.profile_decode["attn"] for layer in self.model_runner.model.layers)
      total_gqa_expand = sum(layer.p_attn.profile_decode["gqa_expand"] for layer in self.model_runner.model.layers)
      total_permute = sum(layer.p_attn.profile_decode["permute"] for layer in self.model_runner.model.layers)
      total_qk = sum(layer.p_attn.profile_decode["qk"] for layer in self.model_runner.model.layers)
      total_softmax = sum(layer.p_attn.profile_decode["softmax"] for layer in self.model_runner.model.layers)
      total_av = sum(layer.p_attn.profile_decode["av"] for layer in self.model_runner.model.layers)
      total_calls = sum(layer.p_attn.profile_decode["calls"] for layer in self.model_runner.model.layers)
      print(
        f"[PROFILE] schedule_steps={schedule_steps} "
        f"avg_step={avg_schedule_step:.6f}s "
        f"total={total_time:.4f}s"
      )
      print(
        f"[PROFILE][ATTN] store_kv={total_store:.4f}s "
        f"get_kv={total_load:.4f}s "
        f"attn={total_attn:.4f}s "
        f"layer_calls={total_calls}"
      )
      print(
        f"[PROFILE][ATTN][DETAIL] "
        f"gqa_expand={total_gqa_expand:.4f}s "
        f"permute={total_permute:.4f}s "
        f"qk={total_qk:.4f}s "
        f"softmax={total_softmax:.4f}s "
        f"av={total_av:.4f}s"
      )
    output = [seq.token_ids for seq in all_seqs]
    if not return_metrics:
      return output

    total_generated_tokens = sum(seq.num_completion_tokens for seq in all_seqs)
    metrics = {
      "prefill_backend": self.model_runner.model.layers[0].p_attn.prefill_backend,
      "decode_backend": self.model_runner.model.layers[0].p_attn.decode_backend,
      "num_seqs": len(all_seqs),
      "prompt_tokens": total_prompt_tokens,
      "generated_tokens": total_generated_tokens,
      "total_time_s": total_time,
      "schedule_steps": schedule_steps,
      "avg_step_ms": (total_time / schedule_steps) * 1000.0 if schedule_steps > 0 else 0.0,
      "throughput_tok_s": total_generated_tokens / total_time if total_time > 0 else 0.0,
    }
    return {
      "outputs": output,
      "metrics": metrics,
    }

      
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
    # print(f'Decoding all_tokens:{all_tokens}')
    return self.tokenizer.decode(all_tokens, skip_special_tokens=True)
