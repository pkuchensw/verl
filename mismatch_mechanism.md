# Engine Mismatch 下的 Partial Rollout 处理机制

## 研究问题

train 引擎（FSDP，π_train）与 rollout 引擎（sglang，π_rollout）对同一模型存在数值不一致（bf16/fp8 量化、kernel 差异）。在 Fully Async + Partial Rollout 下，一条未完成序列的 suffix 由旧模型生成、权重同步后由新模型续生；若新旧模型分布偏移明显，把旧模型生成的高 mismatch 前缀带给新模型续生会产出低质量样本。

目标：在权重同步时识别并丢弃高 mismatch 的 partial prefix，只改动 rollout 的保存/恢复方式，不改 reward 与 batch 语义。

## 整套机制（两层）

### 1. 在线截断（partial rollout 层面）

- 权重同步时，对每个在途 partial prefix 用新模型重算逐 token logprob，与 rollout 保存的 logprob 比较。
- 首个 `|log_ratio| > τ` 的 token 处回退 `b = max(t* - window, 0)`，新模型从 b 续生，丢弃高 mismatch 后缀。
- 尊重 turn 边界，不回退到上一个 turn。
- 代价：每次同步多一次训练引擎前向（量级与 normal old-logprob 相同，在途多时才显著）。

### 2. 训练前筛选（gate）

- 每步多采几个 prompt 组，用 π_k 算整轨迹 sample LLR（逐 token 累加），超阈值整组回退重生成。
- 保证进入训练集的每个样本都被筛过至少一次。
- 有 anti-starvation 兜底（重生成 N 次仍脏则强制接收）。

## 与导师可讨论的取舍

- **τ 是全局固定阈值**：引擎噪声地板（fp8 更明显）决定下限，调太低会误触发、调太高漏抓真实 drift；可演进为分位数/自适应阈值。
- **截断只改了 rollout 的保存/恢复，不改 GRPO 数学**，适合当作 MLSys 系统层方案来主张。
- **验证**：开启 feature 与关闭对照，对比训练质量与丢弃率。
