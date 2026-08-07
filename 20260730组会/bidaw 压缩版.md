# Bidaw 组会 PPT 压缩版

论文：Bidaw: Enhancing Key-Value Caching for Interactive LLM Serving via Bidirectional Computation-Storage Awareness  
中文题目：Bidaw：交互式 LLM 服务中的双向计算-存储感知 KV 缓存  
链接：[USENIX FAST 2026 PDF](https://www.usenix.org/system/files/fast26-hu-shipeng.pdf)

## 第 1 页：标题页

**一句话正文**：Bidaw 关注的是交互式 LLM 服务中，KV cache 放入 `host memory + SSD` 两层存储后，系统为什么仍然被 I/O 拖慢。

**建议配图**：论文封面截图，或自画 `User -> Compute Engine -> Two-tier Storage -> GPU` 简图。

## 第 2 页：背景问题

**一句话正文**：多轮对话需要复用历史 KV，但 GPU 显存放不下全部历史 KV，因此系统需要把 KV 缓存在两层存储中。

**建议配图**：Figure 2，two-tier storage 架构图。

## 第 3 页：现有方案的性能差距

**一句话正文**：现有两层 KV 缓存方案相比 ideal caching 仍有明显差距，response latency 最高高 3.8x，throughput 最高低 2.0x。

**建议配图**：Figure 3，existing works 和 ideal caching 的性能差距。

## 第 4 页：关键观察

**一句话正文**：interactive workload 同时具有长对话、低时间局部性和 KV loading time 波动大三个特征，导致普通调度和普通淘汰策略都不够用。

**建议配图**：Figure 4 + Figure 6 + Figure 7/8，三张图可以压成一页三栏。

## 第 5 页：Bidaw 总体思路

**一句话正文**：Bidaw 的核心是双向感知：compute 侧感知 storage I/O 代价，storage 侧感知用户下一次访问的可能时间。

**建议配图**：Figure 9，Bidaw 系统总览图。

## 第 6 页：创新 1，I/O-aware Scheduling

**一句话正文**：Bidaw 用 `ready queue / preparing queue` 分离快慢请求，并用 `disk-HRRN = 1 + waiting time / KV size` 优先加载更合适的 KV。

**建议配图**：Figure 10，I/O-aware request scheduling strategy。

## 第 7 页：Scheduling 为什么有效

**一句话正文**：I/O-aware scheduling 避免一个慢 KV load 请求堵住后面的快请求，从而减少 GPU 空转和请求排队时间。

**建议配图**：Figure 11 + Figure 19，前者讲机制，后者讲 queuing time 下降。

## 第 8 页：创新 2，Previous-answer-based Eviction

**一句话正文**：上一轮模型回答越长，用户读完并提出下一轮问题通常越晚，因此 answer length 可以作为下一次 KV 访问距离的预测信号。

**建议配图**：Figure 12，previous answer length 和 weighted reuse distance 的相关性。

## 第 9 页：Eviction 怎么做

**一句话正文**：Bidaw 用 ghost cache 估计不同 reuse distance 区间的 hit potential，再结合上一轮 answer length 选择最低潜力的 KV 进行淘汰。

**建议配图**：Figure 13，reuse distance 区间与 hit potential。

## 第 10 页：创新 3，Storage-efficient Tensor Caching

**一句话正文**：Bidaw 不默认缓存 KV，而是选择“节省计算 / 占用空间”更划算的中间 tensor，以提升缓存空间利用率。

**建议配图**：Figure 14，不同 tensor 的 size、saved FLOPs 和 cost efficiency。

## 第 11 页：核心实验结果

**一句话正文**：Bidaw 在多个模型上显著降低平均响应延迟，OPT-13B 上最高达到 3.58x latency improvement，throughput 最高提升 1.83x。

**建议配图**：Figure 15，总体性能结果。

## 第 12 页：Ablation 与总结

**一句话正文**：ablation 证明 scheduling、eviction 和 tensor caching 分别贡献收益，说明 Bidaw 的提升不是单个技巧碰巧带来的。

**建议配图**：Figure 21，Effects of individual techniques。

## 最终收尾句

Bidaw 的贡献可以概括为：它把 KV caching 从单纯的存储管理问题，变成了计算调度和存储淘汰协同优化的问题。
