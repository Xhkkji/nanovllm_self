from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from nanovllm.config import Config

config = Config(
    model_path="/home/xhk/model/Qwen3-0.6B/",
    max_num_seqs=256,              # 同时处理 256 个请求
    max_num_batched_tokens=16384,  # 每批最多 16384 tokens
    max_model_len=4096,            # 每个请求最长 4096 tokens
    gpu_memory_utilization=0.9,    # 使用 90% 显存
    kvcache_block_size=256,         # PagedAttention 块大小
    device="cuda:0"
)
class llm_engine():
  def __init__(self, tensor_parallel_size=1):
    # self.text = text
    self.tokenizer = AutoTokenizer.from_pretrained('/home/xhk/model/Qwen3-0.6B')
    self.model = AutoModelForCausalLM.from_pretrained("/home/xhk/model/Qwen3-0.6B",
    torch_dtype=torch.float16,
    device_map="auto")
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

        newly_finished = (next_tokens.squeeze(-1) == self.eos_token_id)  # 查看哪个batch序列已经生成结束符
        finished = finished | newly_finished
        # 将finished[false, true, false]加第一维，重构为[[false],[true],[false]],与next_tokens对齐
        next_tokens = torch.where(finished.unsqueeze(1),  # condition: 条件判断
                    torch.tensor(self.eos_token_id).repeat(batch_size, 1),  # x: 条件为True时取这个值
                    next_tokens)  # y: 条件为False时取这个值

        all_tokens = torch.cat([all_tokens, next_tokens], dim=1)  # [batch, seq_len+1]
    return all_tokens

  def decode(self, all_tokens):
    return self.tokenizer.decode(all_tokens)