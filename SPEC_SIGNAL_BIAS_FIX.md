# SPEC: 信号偏向修复 — 消除 long 信号系统性缺失

> **日期**: 2026-05-31
> **状态**: 待实施
> **影响范围**: `app/signal_engine/` 全链路

---

## 1. 问题陈述

过去 7 天（~240 条信号），系统输出 **0 次 long、~50 次 short、~190 次 neutral**。同期 ETH 实际存在反弹上涨行情，系统完全未能捕获做多机会。

**目标分布**（长期均衡参考值，非所有时段硬性目标；强单边行情中分布自然偏向一侧是正常的）:
long : short : neutral ≈ **30 : 30 : 40**

---

## 2. 根因分析

系统中存在 **6 个独立的偏向 neutral/short 的机制**，它们叠加后导致 long 信号被完全抑制：

| # | 机制 | 位置 | 类型 | 影响程度 |
|---|------|------|------|----------|
| 1 | Prompt 显式要求 70% neutral | `llm_prompts.py:32` | Prompt 级别 | ⭐⭐⭐ 致命 |
| 2 | Few-shot 只有 1 个 neutral 示例 | `llm_prompts.py:71-132` | Prompt 级别 | ⭐⭐⭐ 致命 |
| 3 | 失效条件门控 (< 2 条 → neutral) | `llm_prompts.py:42` | Prompt 级别 | ⭐⭐ 严重 |
| 4 | Post-check 强制降级 | `llm_agent.py:206-311` | 验证级别 | ⭐⭐ 严重 |
| 5 | Schema RR ≤ 0 强制降级 | `schemas.py:196-206` | Schema 级别 | ⭐ 中等 |
| 6 | ATR floor 硬拦截 | `service.py:98-127` | 服务端级别 | ⭐ 中等 |

### 2.1 机制详解

**机制 1: Prompt 显式 neutral 偏向**
```
"neutral 是高手的常态输出：70% 的行情不值得出手。信号不明确时 bias=neutral
是正确的交易决策，不是"能力不足"。不要为了输出方向而强行串联弱信号。"
```
这直接告诉 LLM "输出 neutral = 你是高手"，形成强烈的正向激励偏向。

**机制 2: 唯一 few-shot 示例是 neutral**
LLM 从未见过一个合格的 long/short 信号 JSON 长什么样。唯一示例展示的是一个"高手的拒绝交易"。这产生了锚定效应：LLM 学到的是"默认应该拒绝"。

**机制 3: 失效条件门控**
```
"如果凑不出 2 条有实际意义的失效条件，说明信号本身不够强，应考虑 bias=neutral"
```
这给 LLM 另一个走向 neutral 的理由。叠加机制 1 后，即使 LLM 想给方向信号，也会因为这个门控而退缩。

**机制 4: Post-check 强制降级**
`llm_agent.py` 的 `_post_check_signal()` 在以下情况强制降级为 neutral：
- RR 自报与复算偏差 > 5% → 降级
- plan 字段不完整（entry_zone / stop_loss / take_profit 缺失）→ 降级
- take_profit < 2 档 → 降级

即使 LLM 正确判断了方向，任何数学参数的不精确都会导致信号被丢弃。

> **注意**：其中"plan 字段不完整"的降级实际上是冗余的——`schemas.py:159-163` 的
> `_post_validate` 会在 **schema 校验阶段** 就抛出 ValueError（要求 long/short 必须
> 给齐 entry_zone / stop_loss / ≥2 档 take_profit），信号在 `llm_agent.py:118-124`
> 就已经被丢弃为 None，根本到不了 `_post_check_signal()`。但 RR 偏差降级是真实生效的。

**机制 5: Schema RR ≤ 0 强制降级**
`schemas.py:196-206`：如果计算出的 RR ≤ 0，整个信号被强制改为 neutral。这在逻辑上是正确的防御，但它与 post-check 叠加后形成了双重降级。

**机制 6: ATR floor 硬拦截**
`service.py:98-127`：当 15m ATR/price < 0.25% 时，系统完全跳过 LLM 调用，直接返回 neutral。这是唯一的服务端硬覆盖，在低波动时段会过滤掉所有行情。

### 2.2 NarrativeRenderer 的叠加效应

`NarrativeRenderer` 的 `_verdict_capital()` 方法使用"对称怀疑"措辞：
- "被动抬升，主动买盘未跟随 → 更像空头回补 / 诱多嫌疑"
- "价格被动下压，主动卖盘未跟随 → 更像多头止损 / 诱空嫌疑"

虽然这些措辞本身是对称的，但叠加 prompt 的 70% neutral 指令后，LLM 在任何方向上都能找到"怀疑的理由"，从而倾向于不出手。

`_comprehensive_capital_judgment()` 方法也存在方向性偏重：
- 当 net_flow 与 CVD 背离时，只提"警惕被动拉盘 / **诱多**"，没有对称地提诱空
- 这与 `_verdict_capital()` 的措辞不完全一致

---

## 3. 修复方案

### 设计原则

1. **多空严格对称**：long 和 short 的触发条件完全镜像，不存在不对称门槛
2. **信任 LLM 判断**：移除所有 prompt 级别的强制 neutral 机制；服务端仅保留数学自洽校验
3. **最小有效修改**：每个修改都有明确的因果关系，不做"顺便"改动
4. **不引入无关变更**：confidence 字段语义优化、数据库 schema 变更等不相关的改进拆到独立 PR

### 变更范围

本次修复 **只做 prompt 重写 + 验证层松绑 + ATR floor 软化**——这三个是因果直接且低风险的改动。
confidence 三档化等结构性变更拆为 **SPEC_SIGNAL_BIAS_FIX_PART2**（另设文档）。

### 修改清单

#### 3.1 `app/signal_engine/llm_prompts.py` — Prompt 重写（机制 1/2/3）

**变更 A: 删除 neutral 偏向语句（机制 1）**

删除第 32-33 行：
```
5. **neutral 是高手的常态输出**：70% 的行情不值得出手。信号不明确时 bias=neutral
是正确的交易决策，不是"能力不足"。不要为了输出方向而强行串联弱信号。
```

替换为：
```
5. **多空对称判断**：每一轮分析都必须同时评估做多和做空的证据强度。
   当 long 证据显著强于 short（或反之）时输出方向信号；当两者证据均衡时输出 neutral。
   不要预设偏向——neutral 不是默认答案，也不是"高手标志"，只是证据均衡的结果。
```

**变更 B: 删除失效条件门控（机制 3）**

删除第 40-42 行：
```
3. **invalidation_conditions**：≥ 2 条量化失效条件，每条带具体价位 / 阈值
   （如 "1h 收盘跌破 3505"、"funding 转负"）。
   如果凑不出 2 条有实际意义的失效条件，说明信号本身不够强，应考虑 bias=neutral。
```

替换为：
```
3. **invalidation_conditions**：尽可能给出量化失效条件（如 "1h 收盘跌破 3505"、"funding 转负"）。
   数量不设硬性下限——有 1 条写 1 条，有 3 条写 3 条。
```

**变更 C: 删除 few-shot 示例（机制 2）**

删除整个 `FEW_SHOT_EXAMPLES` 列表（第 71-132 行），设为空列表：
```python
FEW_SHOT_EXAMPLES: List[Tuple[str, str]] = []
```

理由：删除示例让 LLM 依靠自身判断，避免被唯一的 neutral 示例锚定。同时，删除比添加双向示例风险更低（添加新示例需要大量调优）。

> **注意**：`llm_client.py:131-134` 中 `for human_text, ai_text in FEW_SHOT_EXAMPLES` 循环
> 在列表为空时自动跳过，`few_shot_messages` 保持为空列表，`*few_shot_messages` 解包无
> 任何影响，因此 **`llm_client.py` 不需要任何改动**。

**变更 D: 更新写法对照中的偏向语**

第 58-62 行的"信号不强时"示例写法调整为对称表述：

```
**信号不强时**——列出双向证据，诚实给出结论：

不要写："CVD 微正 + OI 小幅增加 = 多头力量累积中，短期偏多"
要写："CVD 微正但 net_flow 接近零 = 短期买盘力度不强；OI 变化不显著。
       做多证据：无；做空证据：无。当前资金行为不构成任一方向的有效证据，维持观望。"
```

**变更 E: 更新第 64 行的 few-shot 引导语**

删除：
```
下面通过 1 个 few-shot 示例演示"什么时候不出手"——这是最难学也最关键的能力。
```

因为 few-shot 已清空，此行无意义。

**变更 F: HUMAN_PROMPT 末尾提醒**

第 179 行保持不变（1h/4h 为主方向，5m/15m 仅择时），这个提醒是合理的且与偏向无关。

---

#### 3.2 `app/signal_engine/llm_agent.py` — 移除 Post-check 强制降级（机制 4）

**变更: `_post_check_signal()` 方法改为自动修复 + 日志**

将方法从"检测到问题 → 强制降级 neutral"改为"检测到问题 → 自动修复 + 记录日志"。

`_force_neutral_for_post_check()` 方法的**完整调用点清单**及处置方案：

| # | 调用位置 | 当前触发条件 | 可达性 | 处置 |
|---|----------|-------------|--------|------|
| 1 | 第 222-226 行 | plan 字段不完整（ez=None / sl=None / tp<2） | ❌ 不可达 | **删除整个 if 块**（含早返回） |
| 2 | 第 234-238 行 | risk < 1e-9（entry_mid ≈ stop_loss） | ❌ 不可达 | **删除整个 if 块**（含早返回） |
| 3 | 第 243-252 行 | RR 自报与复算偏差 > 5% | ✅ 可达 | **改为自动修复**（见下文） |
| 4 | 第 254-255 行 | take_profit < 2 档 | ❌ 不可达 | **删除**（死代码） |
| 5 | 第 276-281 行 | hard_issues 非空时最终 fallback | ❌ 不可达 | **删除整个 if 块** |

> **不可达原因**：
> - 调用点 #1：`schemas.py:159-163` 的 `_post_validate` 要求 bias=long/short 时 entry_zone /
>   stop_loss / take_profit 必须非空且 tp≥2，不满足则 `raise ValueError`，信号在
>   `llm_agent.py:123` 被捕获返回 None，永远到不了 `_post_check_signal()`。
> - 调用点 #2：`schemas.py:185-188` 校验 `risk_per_unit <= 1e-9 → raise ValueError`，
>   同理不可达。
> - 调用点 #4：第 222 行的早返回已拦截 `len(tps) < 2`，此检查在早返回之后，不可达。
> - 调用点 #5：唯一可达的 hard_issue 来源（#3 RR 偏差）已改为 auto-fix，
>   清理后 hard_issues 列表永远为空。

**具体改造**：

1. **RR 偏差 > 5%（唯一可达的检查）**：不再降级，改为自动用复算值覆盖 `risk_reward_ratio`，
   同时打一条带结构化标记的 warning 日志（`llm_post_check_rr_autofix=1`），
   便于后续通过日志统计自动修复率，判断 LLM 是否存在系统性 RR 误算问题。
   自动修复后，RR 偏差不再记入 hard_issues，仅记入 notes。

2. **删除以下不可达路径**（它们在 schema 校验阶段已被拦截）：
   - 第 222-226 行：`if ez is None or sl is None or len(tps) < 2:` 早返回块
     > ⚠️ **不可保留"仅日志"版本**：如果保留此检查但移除早返回，后续代码
     > `entry_mid = (float(ez[0]) + float(ez[1])) / 2.0` 在 `ez=None` 时会
     > **TypeError crash**。因此必须整块删除，而非改为 log-only。
   - 第 234-238 行：`if risk < 1e-9:` 早返回块
   - 第 254-255 行：`if len(tps) < 2:` 死代码分支
   - 第 276-281 行：`if hard_issues:` 最终 fallback 块

3. **删除第 257-261 行 `invalidation_conditions < 2` 软检查**：§3.1 变更 B 已将 prompt 中的
   "≥ 2 条"硬要求改为"数量不设硬性下限"，此软检查引用的旧规则已不成立，注释
   "（prompt 要求 ≥ 2 条）"与新 prompt 矛盾，一并删除

4. `_force_neutral_for_post_check()` 方法直接 **删除**（不是"废弃保留"）。
   上表已覆盖其全部 5 个调用点，删除后无悬空引用。

返回值仍然是 `Tuple[TradingSignal, List[str]]`，第二个元素为纯信息性的 notes 列表。

---

#### 3.3 `app/signal_engine/schemas.py` — 简化 Schema 降级（机制 5）

**变更: RR ≤ 0 不再强制降级**

`schemas.py:196-206` 中，当 RR ≤ 0 时当前行为是强制设为 neutral 并清空所有 plan 字段。

修改为：当 RR ≤ 0 时，记录 warning 日志，保留 LLM 原始 bias，但将 `risk_reward_ratio` 设为 `None`（表示无法计算）。同时保留 plan 字段不变。

这尊重了 LLM 的方向判断，同时标记了数学不自洽。

> **注意**：实际中此路径几乎不可达——到达这里意味着 prices 已通过顺序校验
> （`entry_mid < tp1` 且 `risk_per_unit > 1e-9`），因此 `rr = reward / risk > 0`
> 在数学上有保证。唯一触发场景是 LLM 显式设置了负的 `risk_reward_ratio`。
> 但作为防御性代码，修改仍然有意义。

**不变更**: `confidence` 字段保持 `float` 类型不变。三档枚举化拆到独立 PR（SPEC_SIGNAL_BIAS_FIX_PART2）。

**不变更**: `_post_validate` 中 neutral 清空字段的逻辑保留（bias=neutral 时不应有计划）。

> **下游安全性**：`risk_reward_ratio` 类型为 `Optional[float]`（第 98 行），设 `None` 类型安全。
> 但需确认下游消费者能正确处理 long/short 信号中 RR 为 `None` 的情况：
> - `analysis_view.py` — 序列化时 `model_dump()` 将 `None` 输出为 JSON `null`，前端需兼容
> - `email_sender.py` → `send_signal_alert()` — `signal` 对象被完整传入，邮件模板中若
>   直接拼接 `risk_reward_ratio` 需做 `None` 防御（`or "N/A"`）
> - `signals` 表 — 数据库字段允许 NULL，无 DDL 变更
> - `service.py` 的 `_dispatch_signal_email()` — 透传 signal 对象，本身不直接引用 RR 字段，
>   但需确认底层模板已做防御
>
> **⚠️ 实施阻塞项**：`email_sender.py` 中对 `risk_reward_ratio` 的所有引用必须加
> `None` 防御处理（`or "N/A"` / 条件渲染），**验证通过后本节变更方可合并**。
> 此检查不可跳过——long/short 信号中 RR 为 `None` 是合法状态。

**额外清理**：

- **`schemas.py` 第 114 行** `invalidation_conditions` 字段的 `description` 包含 `"（≥ 2 条）"`，
  与 §3.1 变更 B（移除 ≥ 2 硬性要求）矛盾，同步修改为 `"量化失效条件，中文短句"`。

---

#### 3.4 `app/signal_engine/service.py` — ATR floor 改为软提示（机制 6）

**变更: ATR floor 不再跳过 LLM，改为将 ATR 信息注入 prompt**

1. **修改** `_atr_floor_check()` 的返回值语义：当前方法触发时返回硬编码字符串 `"atr_too_low"`，
   无法用于 prompt 注入。修改为返回格式化的中文警告字符串（包含实际 ATR 百分比值），未触发时仍返回 `None`。

   具体变更（`service.py:244-279`）：
   - 方法签名不变：`def _atr_floor_check(self, factors) -> Optional[str]`
   - 第 273-278 行：将 `return "atr_too_low"` 改为 `return f"⚠️ 当前 ATR(15m) 占比极低（{atr_pct:.3f}%），波动率处于不可交易区间。请考虑这是否影响你的方向判断。"`
   - 第 251 行 docstring 更新：返回值说明从 `"atr_too_low"` 改为 `"格式化的中文警告字符串"`

   > **注意**：`_make_atr_floor_neutral_signal()` 方法引用了旧返回值 `"atr_too_low"`（通过
   > `zh_map` 字典映射）。但该方法在本次修改中会被 **删除**，因此无需考虑兼容性。
2. `generate()` 中，当 ATR 偏低时：
   - **不再** 跳过 LLM 调用
   - **不再** 调用 `_make_atr_floor_neutral_signal()`
   - 而是在 `prompt_inputs` 中注入一个额外字段 `atr_floor_warning`：
     ```
     "⚠️ 当前 ATR(15m) 占比极低（{atr_pct:.3f}%），波动率处于不可交易区间。
      请考虑这是否影响你的方向判断。"
     ```
3. `_make_atr_floor_neutral_signal()` 方法直接 **删除**（不再有调用方）。

**注入机制**：在 `LLMAgent._build_prompt_inputs()` 中处理 `atr_floor_warning` 字段。

具体做法：`_build_prompt_inputs()` 接受额外的 `atr_floor_warning: Optional[str] = None` 参数，
当非空时，将警告文本追加到返回 dict 的新键 `atr_floor_warning` 中。

然后在 `HUMAN_PROMPT` 模板中，**插入到末尾"提醒"行之前**（第 177 行 `请基于以上市场状态判断方向` 之后、
第 179 行 `**提醒**` 之前）：

```
{atr_floor_warning}

**提醒**：方向判断以 1h/4h 信号为主；5m/15m 仅用于择时，不要因为短周期强信号就逆着大周期开仓。
```

当 `atr_floor_warning` 为空字符串（无警告）时，模板渲染为空行，不影响 prompt 结构。
将 ATR 警告放在提醒之前，使提醒保持最末位置（LLM 对末尾内容关注度最高）。

调用链变更：
```
service.py _atr_floor_check():  # 修改返回值内容（签名不变）
  # 触发时返回格式化警告字符串，未触发返回 None
  return f"⚠️ 当前 ATR(15m) 占比极低（{atr_pct:.3f}%），..."

service.py generate():
  atr_warning = self._atr_floor_check(factors)  # Optional[str]，格式化警告或 None
  # 不再拦截，继续走 LLM
  llm_result = await self.llm_agent.analyze(symbol, factors, atr_floor_warning=atr_warning)

llm_agent.py analyze():
  prompt_inputs = self._build_prompt_inputs(symbol, factors, atr_floor_warning=atr_floor_warning)

llm_agent.py _build_prompt_inputs():
  # 原有逻辑不变，末尾追加：
  result["atr_floor_warning"] = atr_floor_warning or ""
```

**成本影响**：移除 ATR floor 硬拦截后，低波动期也会调用 LLM。按 30 分钟节流间隔估算，
低波动时段（假设每天 8 小时 ATR < 阈值）增加约 16 次 LLM 调用/天。
以 DeepSeek 定价估算，日增成本可忽略。

**额外清理**（ATR floor 改为软提示后，以下代码 / 状态变为 dead code，必须同步清理）：

| 清理项 | 位置 | 说明 |
|--------|------|------|
| `_last_atr_floor_reason` 实例变量 | `service.py:65` | 用于 ATR 边沿触发日志的状态追踪字典；早返回删除后无更新方，变为死状态。**删除字段声明 + 所有引用** |
| ATR 边沿触发日志块 | `service.py:99-127` | 包含状态变更 INFO/DEBUG 分级 + "退出边沿"日志。早返回删除后此块整体失去意义。**整块删除**，替换为 `atr_warning = self._atr_floor_check(factors)` + 可选 `logger.debug` |
| `_make_atr_floor_neutral_signal()` | `service.py:282-309` | 构造 ATR floor neutral 占位信号。早返回删除后无调用方。**整方法删除** |
| `source` 的 `"atr_floor:"` 前缀匹配 | `service.py:555-556` | `_run_loop()` 中 `source.startswith("atr_floor:")` 分支。ATR floor 不再生成该前缀的 source，此分支永不到达。**删除** |
| `"跳过 LLM 调用"` 日志文案 | `service.py:108` | 原日志 `"ATR floor 触发：跳过 LLM 调用"` 已不适用。如保留 DEBUG 日志，改为 `"ATR floor warning 注入 prompt"` |

> **注意**：`_atr_floor_check()` 返回值从硬编码 `"atr_too_low"` 改为包含实际 ATR 百分比
> 的格式化字符串后，**不能再用于字符串相等比较做边沿判断**（ATR 每轮微变 → 每次 `!=`）。
> 边沿触发日志应改为布尔判断（`is None` vs `is not None`）或直接删除（见上表）。

---

#### 3.5 `app/signal_engine/narrative_renderer.py` — 重写为对称证据模式

**变更: 将"对称怀疑"措辞改为"多空证据对称呈现"**

核心改动在以下方法：

**`_verdict_capital()`（第 657-702 行）** — 资金流判断（最关键）

将所有分支从"单方面结论 + 怀疑"改为"做多证据 / 做空证据对称呈现"。完整修改清单：

**OI 四象限**（第 677-684 行，当前已部分对称，措辞微调）：

| 条件 | 当前措辞 | 目标措辞 |
|------|---------|---------|
| `oi_rel=uptrend, oi>0` | "OI 与价格同步上升 = 多头真实建仓中" | **不变**（已对称） |
| `oi_rel=uptrend, oi<0` | "价格上行但 OI 减仓 = 空头平仓推升" | **不变**（已对称） |
| `oi_rel=downtrend, oi>0` | "OI 与价格同步下降 = 空头真实建仓中" | **不变**（已对称） |
| `oi_rel=downtrend, oi<0` | "价格下行但 OI 减仓 = 多头平仓砸盘" | **不变**（已对称） |

**net_flow + CVD 四象限**（第 686-700 行，需改为证据双列模式）：

| 条件 | 当前措辞 | 目标措辞 |
|------|---------|---------|
| `nf>0, cvd>0` | "主动买盘真实，无被动拉盘嫌疑" | "做多证据：net_flow↑ + CVD↑ = 主动买盘真实；做空证据：无" |
| `nf<0, cvd<0` | "主动卖盘真实，无被动砸盘嫌疑" | "做多证据：无；做空证据：net_flow↓ + CVD↓ = 主动卖盘真实" |
| `nf>0, cvd<0` | "价格被动抬升，主动买盘未跟随 → 更像空头回补 / 诱多嫌疑" | "做多证据：无（价格上行但主动买盘未跟随，非真实推升）；做空证据：被动拉盘 = 上涨持续性存疑，可能为空头回补" |
| `nf<0, cvd>0` | "价格被动下压，主动卖盘未跟随 → 更像多头止损 / 诱空嫌疑" | "做多证据：被动砸盘 = 下跌持续性存疑，可能为多头止损回补；做空证据：无（价格下行但主动卖盘未跟随，非真实推跌）" |

**`_comprehensive_capital_judgment()`（第 704-741 行）** — 多周期资金综合判断（逻辑 + 措辞双重修复）

**问题**：当前代码只收集 `nf_dir > 0 and cvd_dir < 0`（被动拉盘/诱多），完全忽略了
`nf_dir < 0 and cvd_dir > 0`（被动砸盘/诱空），导致"诱空"永远不出现在综合判断中。

**代码逻辑变更**（第 727-728 行）：

当前：
```python
if nf_dir > 0 and cvd_dir < 0:
    deception_tfs.append(tf)
```

改为双向收集：
```python
if nf_dir > 0 and cvd_dir < 0:
    deception_tfs_bull.append(tf)   # 被动拉盘 / 诱多方向背离
elif nf_dir < 0 and cvd_dir > 0:
    deception_tfs_bear.append(tf)   # 被动砸盘 / 诱空方向背离
```

**措辞变更**（第 731-735 行）：

当前（仅处理单一列表，措辞偏向"诱多"）：
```python
if deception_tfs:
    return (
        f"{'/'.join(deception_tfs)} 出现 net_flow 与 CVD 背离 → "
        "警惕被动拉盘 / 诱多，方向可信度下降"
    )
```

改为（双向合并返回——多 timeframe 可能同时存在两种背离，`if/if` 顺序返回会丢弃后者）：
```python
parts = []
if deception_tfs_bull:
    parts.append(
        f"{'/'.join(deception_tfs_bull)} net_flow↑ 与 CVD↓ 背离 → "
        "做多证据：无（价格上行但主动买盘未跟随）；"
        "做空证据：被动拉盘 = 上涨非资金推动，方向可信度下降"
    )
if deception_tfs_bear:
    parts.append(
        f"{'/'.join(deception_tfs_bear)} net_flow↓ 与 CVD↑ 背离 → "
        "做空证据：无（价格下行但主动卖盘未跟随）；"
        "做多证据：被动砸盘 = 下跌非资金推动，方向可信度下降"
    )
if parts:
    return "；".join(parts)
```

同时将方法签名中的 `deception_tfs` 变量初始化为两个独立列表：
```python
deception_tfs_bull: List[str] = []   # 替代原来的 deception_tfs
deception_tfs_bear: List[str] = []   # 新增
```

**`_liquidation_overall_verdict()`（第 798-830 行）** — 爆仓判断

当前措辞本身是对称的（多头爆仓 → 空头回补 / 空头爆仓 → 多头平仓），保持不变。

> **注意**：本次修改 **不涉及** `_verdict_liquidity()`（不存在此方法）和 `render_liquidity()`
> 的主体逻辑。流动性段（第 6 段）的措辞已基本对称（"上方真空 → 追多需谨慎" /
> "下方真空 → 追空止损需放宽"），无需改动。

---

## 4. 文件变更汇总

| 文件 | 变更类型 | 变更内容 |
|------|----------|----------|
| `app/signal_engine/llm_prompts.py` | 重写 | 删除 70%neutral 语句；删除失效条件门控；清空 few-shot；更新写法对照；HUMAN_PROMPT 追加 `{atr_floor_warning}` 占位符 |
| `app/signal_engine/llm_agent.py` | 重构 | post-check 改为 RR 自动修复 + 日志；删除 `_force_neutral_for_post_check()` 及全部 5 个调用点（plan 不完整 / risk<1e-9 / RR 偏差 / tp<2 / hard_issues fallback）；删除 `invalidation_conditions < 2` 软检查；`_build_prompt_inputs()` 接受并透传 `atr_floor_warning` |
| `app/signal_engine/schemas.py` | 小改 | RR≤0 不再强制 neutral，改为保留 bias + RR 设 None；`invalidation_conditions` 字段 description 移除"（≥ 2 条）" |
| `app/signal_engine/service.py` | 修改 | `_atr_floor_check()` 返回值从硬编码 `"atr_too_low"` 改为格式化警告字符串；ATR floor 改为软提示注入 prompt；`analyze()` 签名接受 `atr_floor_warning`；删除 `_make_atr_floor_neutral_signal()`；删除 `_last_atr_floor_reason` 状态追踪 + 边沿触发日志块 + `_run_loop()` 中 `atr_floor:` 前缀匹配 |
| `app/signal_engine/narrative_renderer.py` | 修改 | `_verdict_capital()` 4 象限 net_flow+CVD 改为多空证据双列模式；`_comprehensive_capital_judgment()` 双向背离收集（`deception_tfs_bull` / `deception_tfs_bear`）+ 对称证据措辞 |
| `app/signal_engine/llm_client.py` | **无变更** | `FEW_SHOT_EXAMPLES` 为空时循环自动跳过，无需改动 |

**不变更的文件**：
- `schemas.py` 的 `confidence` 字段（保持 `float`）
- `schema.sql` / 数据库 schema（本次无破坏性变更）
- `analysis_view.py` / API 层（响应格式不变）
- `app/config.py`（ATR floor 阈值配置保留，仍可通过 `decision_min_atr_pct_15m` 设 0 关闭）

---

## 5. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 修改后信号过于激进（过多 long/short） | 中 | 中 | 前三天人工审查每条信号质量；prompt 保留了"证据均衡时 neutral"的安全网 |
| NarrativeRenderer 重写引入 bug | 中 | 高 | 重写后用已有 factor dict 做回归测试，确保 7 个 section 都能正确渲染 |
| ATR floor 改为软提示后低波动市场产生低质量信号 | 低 | 低 | LLM 收到 ATR 偏低警告后仍可能输出 neutral；且低波动信号的 RR 通常较差，实际可参考性低 |
| few-shot 删除后 LLM 输出格式不稳定 | 低 | 中 | pydantic schema 验证兜底；且 DeepSeek 对 JSON schema 指令遵循度较高 |
| prompt 增大导致 token 开销上升 | 低 | 低 | NarrativeRenderer verdict 改写每条增加 ~20 token（4 个 timeframe），总体增量 < 5% |

---

## 6. 验证方案

### 6.1 上线前验证

1. **单元测试**：确保 `_post_check_signal()` 的 RR 自动修复路径正确覆盖 `risk_reward_ratio`
2. **Prompt 干跑**：用最近 5 组 factor dict 手动触发 LLM，检查输出分布（预期至少出现 1 次 long）
3. **NarrativeRenderer 回归**：用硬编码的 factor dict 调用 `render_sections()`，确认 7 个 section 输出格式正确
4. **ATR floor 注入验证**：构造 ATR < 阈值的 factor dict，确认 LLM prompt 中包含 ATR 警告文本

### 6.2 上线后观察

1. **前 3 天**：人工审查每条信号的 reason/risk 字段质量
2. **前 7 天**：统计 long : short : neutral 分布，目标接近 30:30:40
3. **持续**：定期查数据库统计 `SELECT bias, COUNT(*) FROM signals GROUP BY bias`

### 6.3 回滚条件

如果修改后出现以下情况，考虑回滚：

- **任意 50 条信号窗口内，单一方向 > 70%**（如 40+ long 或 40+ short）
- 信号 reason 字段质量显著下降（如出现模板化/重复文本）
- LLM 调用失败率 > 10%（schema 验证失败）
- 连续 5 条 neutral 的 reason 都包含"证据均衡"但市场实际有明显趋势

---

## 7. 实施顺序

按依赖关系和风险排序：

1. **Phase 1: Prompt 重写** — `llm_prompts.py` 删除偏向语 + 清空 few-shot + 更新写法对照 + 追加 `{atr_floor_warning}` 占位符
2. **Phase 2: 叙述层对称化** — `narrative_renderer.py` 改写 `_verdict_capital()` 4 象限 + `_comprehensive_capital_judgment()` 双向背离
3. **Phase 3: 验证层松绑** — `llm_agent.py` post-check 改为 RR 自动修复 + 日志；删除 `_force_neutral_for_post_check()` 及全部 5 个调用点；删除 `invalidation_conditions < 2` 软检查；`_build_prompt_inputs()` 透传 `atr_floor_warning`
4. **Phase 4: Schema 松绑** — `schemas.py` RR≤0 不再强制 neutral + `invalidation_conditions` description 移除"（≥ 2 条）"
5. **Phase 5: 服务端松绑** — `service.py` ATR floor 改为软提示 + 删除 `_make_atr_floor_neutral_signal()` + 删除 `_last_atr_floor_reason` 状态追踪 + 删除边沿触发日志块 + 删除 `_run_loop()` 中 `atr_floor:` 前缀匹配

> **部署要求**：Phase 1-2 必须作为原子提交一起部署（prompt + 叙事层共同影响 LLM 行为）。
> Phase 3-5 理论上可独立部署，但建议也合入同一次提交，避免中间状态不一致。

---

## 8. 不在本次范围内的内容

- **confidence 三档化**：拆为独立 PR（SPEC_SIGNAL_BIAS_FIX_PART2），需要同步修改数据库 schema、API 序列化、日志格式化、few-shot 示例等多处
- 数据采集层排查（后续可单独调查 liquidation long 数据完整性）
- 回测框架
- 信号分布监控告警
- LLM 模型切换
- 持仓周期调整
