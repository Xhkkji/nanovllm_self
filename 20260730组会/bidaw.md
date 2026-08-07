# Bidaw 组会汇报稿（12 页版）

论文：Bidaw: Enhancing Key-Value Caching for Interactive LLM Serving via Bidirectional Computation-Storage Awareness  
链接：[USENIX FAST 2026 PDF](https://www.usenix.org/system/files/fast26-hu-shipeng.pdf)

这份文档按 12 页 PPT 来写，每页都给出：

1. PPT 上该放什么文字
2. 建议配什么图
3. 这一页要讲出的结论

建议总时长：约 24-26 分钟。每页时间不用卡死，前 4 页把问题讲清楚，后面方法和实验就会顺很多。

**时间分配建议**

- 第 1 页：1 分钟
- 第 2-4 页：6-7 分钟
- 第 5-10 页：12-14 分钟
- 第 11-12 页：5-6 分钟

---

## 第 1 页：标题页

**PPT 文案**

- 标题：Bidaw: Enhancing Key-Value Caching for Interactive LLM Serving via Bidirectional Computation-Storage Awareness
- 副标题：交互式 LLM 服务中的双向计算-存储感知 KV 缓存
- 论文来源、汇报人、日期

**建议配图**

- 论文封面截图，或一张简洁的 `user -> compute engine -> two-tier storage -> GPU` 流程图

**讲解重点**

- 开场一句话：这篇工作在解决“KV cache 放进两层存储之后，为什么系统还是慢”。

**讲稿（约 1 分钟）**

大家好，我今天汇报的论文是 Bidaw，题目可以翻译成“交互式 LLM 服务中的双向计算-存储感知 KV 缓存”。这篇论文关注的不是普通离线推理，而是多轮对话这种 interactive serving 场景。它的问题很具体：为了避免反复重算历史上下文，我们会缓存历史 KV；但 GPU 显存放不下，就会把 KV 放到 host memory 和 SSD 组成的两层存储里。问题是，放进去之后系统并没有自动变快，反而会被 KV loading 卡住。Bidaw 的核心就是让 compute 侧和 storage 侧互相知道对方的状态，从而把这个 I/O 瓶颈降下来。

---

## 第 2 页：问题背景

**PPT 文案**

- interactive LLM serving 是多轮对话，上一轮生成的 KV 要留给下一轮继续用。
- GPU 显存不够，历史 KV 往往放在 `host memory + SSD` 的两层存储里。
- 这样虽然扩大了容量，但 KV loading 会进入关键路径。
- 核心问题：瓶颈不在算力本身，而在 KV 从存储层到 GPU 的搬运效率。

**建议配图**

- 论文 Figure 2：two-tier storage 架构图
- 或者你自己重画一个更简洁的系统图

**讲解重点**

- 强调“KV loading 在 critical path 上”，后面的优化都围绕这个点展开。

**讲稿（约 2 分钟）**

先看背景。交互式 LLM 服务和单轮请求不一样，它是连续多轮的。第 0 轮用户问问题，模型回答之后会生成这一轮的 KV；第 1 轮再来时，为了保持上下文一致，模型需要用到前面轮次的 KV。如果这些 KV 不缓存，每一轮都要把历史上下文重新算一遍，轮数越多，冗余计算越严重。

但问题是 GPU 显存有限，不可能长期保存所有用户、所有轮次的历史 KV。所以已有工作会把历史 KV 放到两层存储：快的一层是 host memory，慢但容量大的一层是 SSD。这样容量问题解决了，但每次请求来时，KV 需要从两层存储加载到 GPU，KV loading 就进入了请求的关键路径。也就是说，请求不是只等 GPU 计算，还要等 KV 搬运。

这一页要记住的点是：Bidaw 不是在优化模型算子，而是在优化“历史 KV 怎么被及时搬到 GPU”。

---

## 第 3 页：现有方案为什么不够

**PPT 文案**

- 现有两层缓存方案和 ideal caching 之间还有明显差距。
- 论文测到：response latency 最多高 3.8x，throughput 最多低 2.0x。
- 这说明问题不是“有没有缓存”，而是“缓存系统有没有和调度联动起来”。
- 代表系统：vLLM、CachedAttention、FlashGen。

**建议配图**

- 论文 Figure 3：existing works vs ideal caching 的性能差距图

**讲解重点**

- 这一页只做定调，不进入方法细节。

**讲稿（约 2 分钟）**

论文先做了一个很关键的对比：如果所有 KV 都能从 host memory 里加载，也就是 ideal caching，系统性能会怎样；再和现有的两层缓存方案比较。结果差距很明显，现有方案的 response latency 最高可以高 3.8 倍，throughput 最高可以低 2 倍。

这说明一个问题：不是说“我有 KV cache”就够了。CachedAttention、FlashGen 这类方法已经在做两层 KV 缓存，但它们仍然没有接近理想情况。根因是 compute engine 和 storage system 基本是割裂的：compute 侧按自己的请求队列调度，不太关心这个请求的 KV 从哪里来、要加载多久；storage 侧做 eviction 时，也主要看自己的历史访问信息，不知道用户接下来什么时候可能回来。

所以这篇论文的切入点是：把计算调度和存储淘汰联动起来，而不是各管各的。

---

## 第 4 页：工作负载特征

**PPT 文案**

- 多轮对话持续时间长，KV 会在系统中保留很久。
- KV 访问间隔大，temporal locality 差，说明 eviction 不能只看历史访问顺序。
- 不同请求的 KV 大小差异大，导致 load time 差异也很大，说明 scheduling 不能只按 FCFS。

**建议配图**

- 论文 Figure 4：对话持续时间和轮次分布
- 论文 Figure 6：weighted reuse distance 和 hit rate
- 论文 Figure 7 / Figure 8：KV loading time 波动和 loaded KV size 分布

**讲解重点**

- 这页的作用是证明：这是一个“长尾 I/O + 低局部性 + 大波动”的场景。

**讲稿（约 3 分钟）**

接下来论文分析了真实 interactive conversation workload。这里不要陷入图的细节，抓三个观察就够了。

第一个观察是对话很长。很多用户不是问一轮就走，而是持续多轮交互。论文里的 workload 平均有二十多轮对话。这意味着一个用户的 KV 会在系统里停留很久，只要用户还没结束会话，这些 KV 就有可能再次被访问。

第二个观察是访问局部性差。用户发完一轮问题之后，会读模型回答、理解内容、再组织下一轮问题。在这个间隔里，其他很多用户的请求会进来。所以从缓存视角看，同一个用户 KV 的两次访问之间，会夹杂大量其他 KV 访问。Figure 6 用 weighted reuse distance 表达这个现象，很多访问距离已经超过 performance layer 容量，所以普通 LRU/FIFO 很容易做错。

第三个观察是 KV loading time 波动很大。一方面，有些 KV 在 host memory，加载快；有些在 SSD，加载慢。另一方面，不同用户历史长度不一样，KV size 差异也很大。所以两个相邻到来的请求，I/O 代价可能完全不同。这就解释了为什么 FCFS 这类简单调度会出问题：先来的请求未必是适合先算的请求。

这三个观察对应到后面两个设计：loading time 波动大，所以需要 I/O-aware scheduling；局部性差，所以 eviction 需要新的预测信号。

---

## 第 5 页：总体方案

**PPT 文案**

- Bidaw = bidirectional computation-storage awareness。
- compute 侧：知道请求的 KV 在哪一层、大小多少，决定谁先上 GPU。
- storage 侧：知道上一轮回答长度，推测下一次访问什么时候来。
- 额外优化：缓存更省空间、但仍能节省计算的历史 tensor。

**建议配图**

- 论文 Figure 9：系统总览图

**讲解重点**

- 一定要把“bidirectional”讲出来。它不是单向优化，而是 compute 和 storage 互相喂信息。

**讲稿（约 2 分钟）**

Bidaw 的总体设计可以概括成一个词：双向感知。Figure 9 里可以看到 compute engine 和 two-tier storage 之间不再只是简单地 load 和 evict，而是互相传递信息。

第一条方向是 storage 到 compute。compute engine 在调度请求时，会知道每个请求的 KV 在哪一层：是在 performance layer，还是已经落到 capacity layer；同时也知道 KV size 大概多大。这个信息用来决定哪些请求先进入 GPU，避免慢 I/O 请求堵住快请求。

第二条方向是 compute 到 storage。storage 做 eviction 时，不只看过去的 KV 访问记录，还会拿到 compute engine 生成的上一轮模型回答长度。这个回答长度用来预测用户下一轮什么时候会回来，从而判断哪些 KV 更值得留在 performance layer。

第三个模块是 history cacher，它不是本文最核心的点，但可以进一步提升空间效率：不一定缓存 KV 本身，而是缓存一个更划算的中间 tensor。

---

## 第 6 页：创新 1，I/O-aware scheduling

**PPT 文案**

- 把请求分成两个队列：
  - `ready queue`：KV 已在 performance layer，可直接进 GPU
  - `preparing queue`：KV 还在 capacity layer，先搬上来再进入 ready queue
- 对 ready queue，按 FCFS 保持公平。
- 对 preparing queue，用 `disk-HRRN` 排序。
- `disk-HRRN = 1 + waiting time / KV size`
- 目标：先让小 KV、等得久的请求更快完成加载，减少 GPU 被慢 I/O 阻塞。

**建议配图**

- 论文 Figure 10：I/O-aware request scheduling strategy
- 如果你自己画图，建议画成上下两个队列，箭头表示 promotion

**讲解重点**

- 这页的核心不是“排队”，而是“把快请求和慢请求隔离开，避免慢请求堵住整个 GPU”。

**讲稿（约 3 分钟）**

先看第一个创新：I/O-aware scheduling。传统调度的问题是，它主要看请求到达顺序，或者看 GPU memory 是否能放下请求，但没有充分考虑 KV loading 代价。Bidaw 的做法是把请求分成两个队列。

如果请求的 KV 已经在 performance layer，也就是 host memory 中，那么它进入 ready queue。这类请求的 KV load 相对快，可以尽快进入 GPU 计算。如果请求的 KV 还在 capacity layer，也就是 SSD 中，那么它先进入 preparing queue。它要先把 KV 从 SSD 搬到 host memory，搬完后再晋升到 ready queue。

ready queue 里仍然使用 FCFS，保证公平性。真正有意思的是 preparing queue：因为 SSD 读取慢，而且不同请求 KV size 差别很大，所以 Bidaw 用 disk-HRRN 来决定先搬哪个 KV。公式是 `1 + waiting time / KV size`。它的直觉很朴素：KV 小的请求更容易快速搬完，应该优先；但等待时间越长，优先级也会逐渐升高，避免大 KV 请求一直饿死。

所以这一页的重点不是发明了多复杂的调度算法，而是把 storage I/O 代价引入到 compute scheduling 里。

---

## 第 7 页：创新 1 的效果与直觉例子

**PPT 文案**

- 如果按 FCFS，最早来的慢请求会卡住后面所有本来可以快算的请求。
- Bidaw 的做法是：先跑能快速完成 I/O 的请求。
- 这样 GPU 不会因为一个慢 KV load 空转，系统整体等待时间也更短。
- 这一步主要减少 queueing time，而不是改变模型输出。

**建议配图**

- 论文 Figure 11：I/O-oblivious 和 I/O-aware 的对比图
- 论文 Figure 19：request queuing time CDF

**讲解重点**

- 这页把“机制图”和“结果图”绑在一起讲，听众更容易接受这个 scheduler 为什么有效。

**讲稿（约 2 分钟）**

这一页用 Figure 11 的例子会很好讲。假设 request 1 最早到，但它的 KV 在 SSD，而且 size 很大；后面的 request 3、4、5 的 KV 都在 host memory，加载很快。如果按 FCFS，request 1 会先被调度，但 GPU 必须等它的 KV 从 SSD 搬完，这期间后面那些本来可以很快跑的请求也被堵住。

Bidaw 的做法是把 request 1 放到 preparing queue，让它慢慢准备；同时 ready queue 中已经能快速加载的请求先上 GPU。这样 GPU 不会因为一个慢 I/O 请求空转。

Figure 19 是这个设计的直接结果：I/O-aware scheduler 明显降低了请求进入 GPU 计算前的排队时间。这里可以强调一点：这个调度不会改变单个请求内部 token 生成的计算顺序，所以它是 lossless 的，不会影响模型回答的准确性。

---

## 第 8 页：创新 2，previous-answer-based eviction 的直觉

**PPT 文案**

- 传统 eviction 只看过去 KV 访问历史，忽略了用户下一次什么时候会回来。
- 在 interactive LLM serving 里，下一次 KV 访问间隔和上一轮模型回答长度相关。
- 回答越长，用户阅读、理解、思考和追问需要的时间通常越长。
- 因而：上一轮 answer length 可以作为下一次 access distance 的代理信号。

**建议配图**

- 论文 Figure 12：不同时间段、不同 arrival rate 下，weighted reuse distance 和 previous answer length 的关系

**讲解重点**

- 这页要讲清楚一件事：上一轮回答长度不是“绝对因果”，而是一个有统计相关性的预测信号。

**讲稿（约 3 分钟）**

第二个创新是 previous-answer-based eviction。这里的问题是：performance layer 放不下所有 KV，必须把一部分 KV evict 到 SSD。普通做法会看历史访问，比如 LRU/FIFO，或者看当前等待队列。但 interactive serving 的 KV 访问局部性很差，只看历史很容易判断错。

Bidaw 提出的关键观察是：上一轮模型回答长度和下一次 KV 访问距离之间有相关性。直觉也比较自然：如果模型上一轮回答很短，用户可能很快就读完并继续追问；如果回答很长，用户读完、理解、思考和组织下一轮问题需要更久。在这段时间里，系统会服务很多其他用户请求，所以这个用户 KV 的 weighted reuse distance 会变大。

Figure 12 展示的是不同时间段、不同 arrival rate 下，这个相关性仍然存在。这里不要讲成“回答长一定回来晚”，它不是确定规则，而是一个统计预测信号。Bidaw 的价值就在于把这个 compute 侧天然产生的信息用于 storage eviction。

---

## 第 9 页：创新 2 的具体做法

**PPT 文案**

- 先统计历史 trace 中不同 weighted reuse distance 区间的 hit potential。
- 用 ghost cache 模拟 optimal eviction，估计各个距离区间的命中概率上界。
- 在线时记录用户上一轮 answer length，收缩下一次访问可能落在哪些区间。
- 对每个用户的下一次 KV access 计算 overall hit potential，最差的先 evict。

**建议配图**

- 论文 Figure 13：不同 weighted reuse distance 范围的 hit rate 示例
- 如果你自己画图，建议画“距离分桶 + 命中概率 + answer length 约束”的流程图

**讲解重点**

- 这里要传达一个关键点：即使 reuse distance 很大，也不代表没有 hit potential，所以不能粗暴 FIFO/LRU。

**讲稿（约 3 分钟）**

有了上一轮 answer length 之后，Bidaw 还需要把它变成具体的 eviction 决策。它大概分四步。

第一步，把 weighted reuse distance 分桶。小距离的访问一般更容易命中，极大距离的访问基本没机会命中，中间还有一段 promising range，也就是虽然距离已经比较大，但仍然有一定命中潜力。

第二步，维护一个 ghost cache。ghost cache 不真正存数据，而是在后台用过去的 trace 模拟 optimal eviction，估计不同 reuse distance bucket 的 hit potential。这里的 optimal eviction 是为了估计上界，不是在线系统真的知道未来。

第三步，在线服务时记录每个用户上一轮模型回答长度，用它预测下一次访问的 weighted reuse distance 下界。这个下界可以排除一些不可能太近的 bucket。

第四步，对每个用户计算下一次访问的 overall hit potential。hit potential 越低，说明把它留在 performance layer 的收益越小，就越适合被 evict 到 SSD。

这页可以把 Figure 13 讲成一个直觉：reuse distance 大不等于完全没价值，中间有一段仍然可能命中。所以 Bidaw 不是简单地按远近淘汰，而是在估计“留下它还有多大可能带来 hit”。

---

## 第 10 页：创新 3，storage-efficient tensor caching

**PPT 文案**

- 不是所有历史 tensor 都值得缓存。
- 不同 tensor 的“节省计算 / 占用空间”比值差别很大。
- Bidaw 选的是 cost efficiency 更高的 tensor，而不是默认只缓存 KV。
- 缓存后再把它变换回 KV，达到更好的空间-计算折中。

**建议配图**

- 论文 Figure 14：不同 tensor 的大小、节省计算量和 cost efficiency

**讲解重点**

- 这部分是工程增强，不要讲太久。它不是主贡献，但能解释为什么系统还能再往前推一点。

**讲稿（约 2 分钟）**

第三个创新可以讲短一点，它更像工程增强。已有 KV cache 系统默认缓存历史 KV，因为下一轮推理要用 KV。但论文注意到，在 transformer 计算过程中不只产生 KV，还会产生很多中间 tensor。这些 tensor 之间有的可以转换成 KV，而且它们的大小和能节省的计算量不一样。

所以 Bidaw 定义了一个 cost efficiency，也就是“节省计算量 / 占用存储空间”。如果某个中间 tensor 占用更小，同时离 KV 很近、转换成本也不高，那么缓存它可能比直接缓存 KV 更划算。

Figure 14 展示了不同 tensor 的空间、节省 FLOPs 和 cost efficiency。这里的结论是：Bidaw 会选择更高 cost efficiency 的 storage-efficient tensor，从而让 performance layer 能容纳更多历史信息。它不是本文最核心的设计，但能进一步提升整体效果。

---

## 第 11 页：实验设置 + 核心结果

**PPT 文案**

- 模型：OPT-6.7B、Qwen-7B、OPT-13B、Qwen-14B、OPT-30B。
- 对比：vLLM、CachedAttention、FlashGen、ideal upper bound。
- 工作负载：作者自建 interactive conversation workload + public ShareGPT workload。
- 硬件：A800 GPU、200GB host memory、1.5GB/s SSD。
- 主结果：Bidaw 在多个模型上显著降低平均响应延迟，在 OPT-13B 上最高可达 3.58x improvement，throughput 最高可达 1.83x。

**建议配图**

- 论文 Figure 15：不同模型下的平均响应延迟

**讲解重点**

- 这一页把实验设置和主结果合在一起，节省一页空间。
- 讲结果时不要逐模型展开，选一个代表模型说清楚，再补一句“其他模型同趋势”。

**讲稿（约 3 分钟）**

实验部分我会重点讲主结果和 ablation。设置上，论文用了多个模型，包括 OPT 和 Qwen 的不同规模；baseline 包括 vLLM、CachedAttention、FlashGen，以及一个 ideal upper bound，也就是假设所有 KV 都能从 host memory 加载。硬件环境是单张 80GB A800、200GB host memory 和 1.5GB/s SSD。

Figure 15 是最重要的结果图。横轴是 user arrival rate，纵轴是 average response latency。可以看到，当 workload pressure 增大时，vLLM 因为要重算历史上下文，延迟会上升很快；CachedAttention 和 FlashGen 虽然缓存 KV，但由于大量 KV 需要从 SSD 加载，延迟也会涨。Bidaw 的曲线整体更低，而且能支撑更高的 user arrival rate。

数字上，OPT-13B 上最高有 3.58 倍 latency improvement，throughput 最高提升 1.83 倍。这里我不会逐个模型拆开讲，因为趋势基本一致。核心结论是：Bidaw 的收益不是只出现在某一个模型上，而是在不同模型规模下都能改善 interactive serving 的整体性能。

---

## 第 12 页：Ablation 与总结

**PPT 文案**

- 只加 I/O-aware scheduling，就能明显降低延迟。
- 再加 previous-answer-based eviction，吞吐继续提升。
- 再加 storage-efficient tensor caching，还有额外收益。
- 结论：三项设计分别打中了不同瓶颈，整体不是堆料。

**建议配图**

- 论文 Figure 21：Effects of individual techniques

**讲解重点**

- 这页作为收尾非常合适：Figure 15 证明整体效果好，Figure 21 证明效果不是单个技巧碰巧带来的，而是三个模块分别打中了不同瓶颈。

**讲稿（约 3 分钟）**

最后看 ablation。Figure 15 只能说明 Bidaw 整体效果好，但还不能说明到底为什么好。Figure 21 的作用就是拆开三个模块看贡献。

第一个版本只加入 I/O-aware scheduling，可以明显降低延迟。这说明前面说的 I/O-induced blocking 确实是主要瓶颈之一。第二步再加入 previous-answer-based eviction，吞吐继续提高，说明用上一轮 answer length 做 eviction 信号确实能改善 performance layer 的有效命中。第三步再加入 storage-efficient tensor caching，还有进一步收益，说明缓存对象的选择也有价值。

所以这篇论文的结论不是“某个小技巧让系统快了”，而是三个模块分别对应三个问题：scheduling 解决慢 I/O 阻塞，eviction 解决低命中率，tensor caching 提升空间效率。最后可以用一句话收束：Bidaw 把 KV caching 从单纯的存储管理问题，变成了计算调度和存储淘汰协同优化的问题。

---

## 哪些实验最重要

### 必讲

1. Figure 15：总体 latency / throughput，主结果。
2. Figure 21：ablation，证明三项设计都有效。
3. Figure 19：queuing time，直接支撑 scheduler。

### 可以讲，但不是第一优先级

1. Figure 18：miss rate，支撑 eviction 机制。
2. Figure 17：ShareGPT 泛化，证明不是只对私有 trace 有效。
3. Figure 22：tail latency，作为补充结果。

### 可以弱化

1. Figure 16：host memory size sensitivity。
2. Figure 20：overhead analysis。
3. Figure 12 / Figure 13 / Figure 14：方法支撑图，主要用来辅助讲直觉。

## 推荐的汇报顺序

1. 问题有多严重。
2. 为什么现有方法不够。
3. 两个关键观察。
4. Bidaw 总体框架。
5. 创新 1：I/O-aware scheduling。
6. 创新 2：previous-answer-based eviction。
7. 创新 3：storage-efficient tensor caching。
8. 实验主结果。
9. ablation。
10. 总结。

## 可以直接念的收尾句

Bidaw 不是单纯把 KV cache 搬到更大存储里，而是把请求调度和缓存淘汰一起联动起来：compute 侧知道 I/O 代价，storage 侧知道用户下一次什么时候大概率会回来。它的主要价值在于减少 I/O 阻塞、提高命中率，所以整体 latency 和 throughput 都能稳定改善。
