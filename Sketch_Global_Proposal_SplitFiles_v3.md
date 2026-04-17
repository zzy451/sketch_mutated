# Research Proposal: Mechanism-First Global Plan for Discovering and Promoting Novel Sketches (Split-File Implementation Version)

## 1. Background & Motivation

本方案以 `sketch方案new.txt` 为**主体主线**，以我们前一轮关于 primitive / runtime / proposal 的方案为**补充与辅助**，并且把**直接施工对象**明确改成你当前这套分文件工程：

- `mutate_cmsketch_refactored.py`（入口脚本）
- `common.py`
- `evaluator.py`
- `helpers.py`
- `llm_engine.py`
- `evolution.py`
- `cli.py`
- `init_dex_language.py`
- `update_language.py`
- `query_language.py`

也就是说：

- **不再**把 `mutate_cmsketch.py` 写成当前唯一施工对象；
- **改为**把它视作“历史来源映射文件”；
- 当前真正要产出完整可运行代码的实施对象，是上面这套**分文件版本**。
- 三个 primitive set 文件（`init_dex_language.py` / `update_language.py` / `query_language.py`）不再视为外围参考，而是这套分文件工程的**grammar/runtime/reference source-of-truth**。

项目总目标依然不变：

> 不是把系统继续打磨成“更会修补 CMS 变体”的 GP+LLM 工具，
> 而是把它推进成一个能够主动发现、保活、晋升、并最终让新机制反向塑造主搜索的 sketch 程序搜索系统。

因此，这份 split-file version proposal 的总原则是：

1. **主体仍然是三条闭环**，不是 primitive set 本身；
2. **实现路径必须落到分文件代码职责**，而不是抽象讲单文件；
3. **primitive/runtime 升级只能服务主线闭环**，不能喧宾夺主；
4. **最终交付必须是完整可运行的分文件代码**。

### 1.1 当前阶段判断

按照 `sketch方案new.txt` 的阶段定义，项目当前处于：

**“主框架基本成形，但关键创新闭环还没有完全打通”的第五阶段。**

当前已经具备：
- evaluator / AST / legality / semantic repair；
- GP + LLM 混合搜索；
- mechanism schema / family / cluster；
- novelty / irregular novelty / incubator / promote；
- dual champion（exploit best + mechanism best）；
- 真实 univ2_trace / flowid 数据评估与 holdout / stage2 real。

所以当前 proposal 的重点不是“设计新框架”，而是：

- 把现有能力接成闭环；
- 让 mechanism-first 从观测层走向动力层；
- 让分文件工程真正支撑这一目标。

### 1.2 三条最终闭环（本 proposal 的总骨架）

这份 proposal 的所有修改，都必须明确回答它在推进哪一条闭环：

1. **numeric-risk 真正前门化**  
   让 seed / validate / runtime compatibility / gp-child 的风险口径统一，尽早挡掉语义合法但数值爆炸的候选。

2. **novelty 从 family 保活升级成 mechanism-cluster 多峰分叉**  
   让 novelty 不只是“活下来”，而是真正形成多个机制峰，不再快速塌成 dominant family。

3. **mechanism best 真正反哺主搜索**  
   让 mechanism champion 从“观测层记录”进入“动力层反馈”，影响 elite、migration、seed refresh、novelty targeting、candidate ranking。

### 1.3 这份 proposal 不是什么

这份 proposal **不是**：
- 一个推翻现有 split-file 工程的新框架；
- 一个只讲 primitive set 的语法设计文档；
- 一个以复刻 NitroSketch / Cold Filter / CocoSketch 为目标的 imitation plan。

这份 proposal **是**：
- 一份以 `sketch方案new.txt` 为主体的**全局实施方案**；
- 一份按 `common / evaluator / helpers / llm_engine / evolution / cli / init_dex_language / update_language / query_language` 九个核心文件分工的**代码施工图**；
- 一份最终指向“完整可运行分文件代码”的落地计划。

---

## 2. Proposed Methodology

本方案由十个直接施工文件和一个辅助模块组成：

- 七个运行/搜索文件：`mutate_cmsketch_refactored.py`、`common.py`、`evaluator.py`、`helpers.py`、`llm_engine.py`、`evolution.py`、`cli.py`；
- 三个 primitive-set 文件：`init_dex_language.py`、`update_language.py`、`query_language.py`；
- 一个辅助模块：primitive/runtime 升级，仅作为主线闭环的配套层。

### 2.1 Module A: 以 split-file 工程为唯一施工对象

当前直接实施对象必须是：

- `mutate_cmsketch_refactored.py`
- `common.py`
- `evaluator.py`
- `helpers.py`
- `llm_engine.py`
- `evolution.py`
- `cli.py`
- `init_dex_language.py`
- `update_language.py`
- `query_language.py`

#### 文件职责重新明确

**1. `mutate_cmsketch_refactored.py`**
- 仅作为入口脚本；
- 负责把 CLI 参数接到 `evolution.py` 和 `evaluator.py`；
- 不再承载主要逻辑。

**2. `common.py`**
- 统一 primitive registry；
- 把 `init_dex_language.py` / `update_language.py` / `query_language.py` 接成 source-of-truth；
- shared constants / counters / matrix config；
- grammar version 与 primitive tier 开关；
- shared runtime helper；
- primitive reference path auto-discovery。

**3. `evaluator.py`**
- `CMSketchEvaluator`；
- tree -> AST -> simplify / legality / effect summary；
- runtime 执行语义；
- `generate_complete_code` / exported runtime 语义一致性。

**4. `helpers.py`**
- seed bank / seed filter / seed injection helper；
- mechanism metadata / schema / cluster key；
- numeric-risk gate；
- novelty axes / dominant cooldown / mechanism diversity override。

**5. `llm_engine.py`**
- `LLMProposalEngine`；
- seed / repair / novelty / irregular novelty proposal；
- schema materialize / validation；
- primitive dual-source validation（runtime pset + primitive reference files）；
- motif card / primitive catalogue。

**6. `evolution.py`**
- `evolve_cmsketch` 主循环；
- island state；
- novelty incubator / promote；
- migration / elite / champion feedback；
- experiment telemetry。

**7. `cli.py`**
- 参数解析；
- phase / grammar / primitive tier 开关；
- quick run / formal run 入口配置；
- primitive reference file 默认路径注入与 consistency log。

**8. `init_dex_language.py`**
- init_dex 侧 typed primitive set 的 source-of-truth；
- `list_3 / select_hash / hash_salt / hash_on_slice` 等 grammar 定义；
- init layout 输出格式（尤其 `(x, y, z)` 三元组）规范。

**9. `update_language.py`**
- update 侧 typed primitive set 的 source-of-truth；
- `update_count / write_count / update_state / query_count / query_state` 等写入与探针 grammar；
- 与 evaluator/export runtime 对齐的局部参考运行时语义。

**10. `query_language.py`**
- query 侧 typed primitive set 的 source-of-truth；
- `query_date / cnt_rdstate / base_sel / safe_min / safe_max / median3 / sum3` 等 query grammar；
- 与 evaluator/export runtime 对齐的局部参考读取语义。

#### 当前最重要的原则

从这一版开始，proposal 的每一步都必须回答：

- 这一步具体改哪个 split file？
- 改的是哪个函数或哪段职责？
- 是否要求 `init_dex_language.py / update_language.py / query_language.py`、`evaluator runtime`、`exported runtime` 同步？
- 它在推进哪一条闭环？
- 最后会在日志里看到什么变化？

---

### 2.2 Module B: 后半程 LLM 主链持续畅通，但不再当主问题

这部分已经修过，当前不是主战场，但必须纳入持续监控。

#### 具体改动文件

**`common.py` + `cli.py`**
1. 为三个 primitive set 文件建立默认 reference path 发现与注入；
2. 当 `--llm_ref_*_pset_path` 为空时，自动绑定到当前工程中的 `init_dex_language.py` / `update_language.py` / `query_language.py`；
3. 启动时输出 primitive registry / consistency log，避免 reference file 漏绑。

**`llm_engine.py`**
1. 对 seed / repair / novelty / irregular novelty proposal 全链路增加结构化状态码：
   - proposed
   - validated
   - materialized
   - accepted
   - rejected
   - skipped
2. proposal validation 失败时统一返回 reason code。
3. schema materialize 失败时记录：
   - primitive mismatch
   - missing helper
   - runtime incompatibility
   - numeric-risk reject。

**`evolution.py`**
1. 在每轮 generation 日志中输出：
   - `llm_seed_proposed / accepted`
   - `llm_repair_proposed / injected`
   - `llm_novelty_proposed / incubated / promoted`
   - `llm_trigger_skip_reasons`
2. 把 trigger-skip 与 materialize-fail 区分输出。

**`cli.py`**
- 增加 `--log_llm_pipeline` 开关。

#### 这一步推进哪条闭环

这一步不是直接推进三条闭环，而是在**保护 innovation path 的基础通道**，避免后续任何闭环优化因 proposal/materialize 链断裂而失效。

#### 期望结果

- 不再出现“后半程跳过但原因不明”；
- 后续如果链断了，能立刻知道是 validation、materialization、numeric-risk 还是 apply 阶段的问题。

---

### 2.3 Module C: numeric-risk 真正前门化（闭环 1）

这是当前第一优先级。

#### 具体改动文件

**`helpers.py`**
1. 建立统一的 risk verdict 数据结构：
   - `verdict`
   - `allow`
   - `warning`
   - `reject`
   - `hard_reject`
   - `runtime_compatible`
   - `seed_bank_reject`
   - `reason_codes`
2. 把现有 numeric-risk 的 phase profile 收拢成同一套语义，只允许阈值差异，不允许语义差异。
3. 把以下入口统一接入同一风险探针：
   - seed bank build
   - island seed inject
   - llm validate
   - llm novelty / irregular novelty
   - llm repair
   - gp child accept
4. 风险规则拆成：
   - structural illegal（仍由 evaluator/AST 管）
   - numeric unstable（全部归 numeric-risk）

**`evaluator.py`**
1. 对 exported runtime 和 evaluator runtime 增加统一的 numeric telemetry：
   - overflow hits
   - impossible query values
   - runaway write attempts
   - repeated saturation patterns
2. 提供给 helpers 的 probe 接口统一返回这些统计。

**`evolution.py`**
1. generation summary 增加：
   - `risk_reject_by_phase`
   - `risk_warning_but_allowed_by_phase`
   - `risk_reason_topk`
2. 对 warning 通过但实际 eval 爆炸的候选做反查日志。

**`cli.py`**
- 增加 `--log_numeric_risk`、`--strict_numeric_front_gate`。

#### 期望结果

- 语义合法但数值极不稳定的候选更早被挡掉；
- seed / novelty / repair 的污染下降；
- 同一候选不再在不同 phase 得到矛盾 verdict。

#### 通过标准

- `risk_reason_topk` 开始收敛；
- warning-to-accept 率可解释；
- “validate 过了但 runtime 完全炸掉”的比例明显下降。

#### 最小日志验收口

每次修改闭环 1 后，至少必须同时检查下面这些日志字段：

- `risk_reject_by_phase`
- `risk_warning_but_allowed_by_phase`
- `risk_reason_topk`
- `DIAG_CHUNK` 中的 `penalty_avg / query_avg / total_avg`
- `llm validation bottleneck` 与 `runtime/reference conflict accepted by runtime compatibility`

验收标准不是单看 best error，而是：

1. warning / reject 的口径开始收敛；
2. 1e9 量级数值爆炸块明显减少；
3. “warning 通过但后续 runtime 爆炸”的候选比例下降；
4. seed / novelty / repair 三条链上的 numeric-risk verdict 更一致。

---

### 2.4 Module D: seed 去坏保好（当前第二个关键入口）

目标不是更多 seed，而是**提高高质量密度，同时保住机制新颖 seed**。

#### 具体改动文件

**`helpers.py`**
1. 重写 / 收紧 seed filter：
   - exploit-biased quality gate
   - numeric-risk gate
   - canonical-duplicate gate
   - mechanism-distance 保留槽
2. seed bank 从单桶改成三桶：
   - exploit-biased
   - novelty-biased
   - mechanism-gap-filling
3. island 注入前二次过滤：
   - 当前 island best error
   - 当前 island dominant family / cluster
   - 当前 island 最近注入历史
   - 当前 island 缺失机制峰
4. 为“中等性能但机制新”的 seed 建单独保留通道。

**`llm_engine.py`**
- seed proposal 输出时显式附带：
  - estimated mechanism family
  - mechanism distance hint
  - expected risk class

**`evolution.py`**
- 增加：
  - `seed_dropped_bad_quality`
  - `seed_dropped_numeric_risk`
  - `seed_kept_for_mechanism_novelty`
  - `seed_injected_by_bucket`

#### 推进哪条闭环

- 直接服务闭环 1（减少脏候选污染）
- 间接服务闭环 2（保住未来 novelty / mechanism 分叉种子）

#### 期望结果

- 坏 seed 注入减少；
- 机制新颖 seed 不再被一刀切清空；
- 初始种群与中途 seed refresh 的质量密度提高。

#### 三桶 seed 的配额与岛路由

为了避免 seed bank 只在 exploit 或 novelty 一侧极端化，三桶 seed 必须显式配额，并按 island profile 路由：

**建议默认配额**
- exploit-biased：40%
- novelty-biased：35%
- mechanism-gap-filling：25%

**建议默认 island 路由**
- `baseline`：以 exploit-biased 为主，少量 novelty-biased
- `init_explore`：以 novelty-biased 为主，辅以 mechanism-gap-filling
- `update_explore`：novelty-biased 与 mechanism-gap-filling 混合
- `irregular_architecture`：以 mechanism-gap-filling 为主，保留少量 novelty-biased

如果某一桶当前为空，不允许直接把全部预算回填给 exploit-biased，而应优先回填到与该 island profile 更接近的第二选择桶。

---

### 2.5 Module E: novelty 从 family 保活升级到 mechanism-cluster 多峰分叉（闭环 2）

这是当前第二条闭环的主战场。

#### 具体改动文件

**`helpers.py`**
1. novelty 判断单位从单纯 family 推向：
   - mechanism family
   - mechanism cluster
   - novelty axes
2. dominant cooldown 逻辑细化：
   - 不再只按 family 一刀切；
   - 对同 family 不同 mechanism cluster 分开限流；
   - 对稀缺 mechanism axis 提供额外放行权重。
3. mechanism diversity override 重写：
   - 允许更大 distance 的 cluster 绕过 dominant family cooldown；
   - 允许 state-contract / query-contract 不同的机制单独保活。
4. incubator entry / promote gate 统一接 mechanism metadata。

**`evolution.py`**
1. incubator / promote 主逻辑直接用 mechanism-cluster 作为第一排序维度之一；
2. novelty telemetry 增加：
   - `novelty_mechanism_cluster_count`
   - `novelty_family_count`
   - `dominant_family_cooldown_hits`
   - `mechanism_override_passes`
3. novelty budget 按 family-only 改为 family + cluster + axes 混合分配。

**`llm_engine.py`**
- novelty / irregular novelty 的 proposal 模板里显式支持：
  - role
  - handoff
  - state contract
  - query contract
  - lane relation

#### 推进哪条闭环

- 直接推进闭环 2。

#### 期望结果

- novelty 不再只会塌在 dominant family；
- incubator / promote 中出现多个稳定机制峰；
- mechanism cluster 多样性增加。

#### 通过标准

- `novelty_mechanism_cluster_count` 持续非零并增长；
- promote 不再几乎全来自同一 dominant family 邻域。

#### 最小日志验收口

每次修改闭环 2 后，至少必须同时检查：

- `novelty_mechanism_cluster_count`
- `novelty_family_count`
- `dominant_family_cooldown_hits`
- `mechanism_override_passes`
- `novelty_incubated / novelty_promoted`
- `novelty_reject_top3`

验收重点不是仅看 promote 数量，而是：

1. promote 不再几乎全部来自同一 dominant family；
2. 新 cluster 能稳定进入 incubator；
3. family cooldown 对真正新 cluster 的误伤减少；
4. novelty 的多峰结构能够持续几个迁移周期，而不是出现后立刻塌缩。

---

### 2.6 Module F: repair 降权，innovation path 增权

repair 不是去掉，而是从主推进器降为局部修补工具。

#### 具体改动文件

**`helpers.py`**
1. 收紧 repair quality gate：
   - 必须对当前 target 有真实改善；
   - 且不能离当前 island best 太远；
   - 且不能把机制簇直接拉回 canonical basin。
2. 为 repair 候选增加 canonical-basin 吸力惩罚。

**`evolution.py`**
1. repair 预算上限固定，未用完预算自动回流 novelty；
2. winner slot 配额调整：
   - repair slot 下调
   - novelty / mechanism slot 上调
3. 对 mechanism-best lineage 设置单独 survivor quota。

**`llm_engine.py`**
- repair prompt 中显式禁止“纯回到 canonical team”的低价值修补倾向。

#### 推进哪条闭环

- 直接支持闭环 2 和闭环 3；
- 防止系统被重新拉回 repair/CMS path。

#### 期望结果

- repair 仍然有用，但不再一枝独大；
- novelty winner / mechanism winner 的进入主种群比例提升。

#### 最小日志验收口

每次修改 repair 配额或质量门后，必须同时对比：

- `repair_injected` vs `novelty_injected`
- repair candidate reject reasons
- duplicate canonical reject 占比
- 当前 run 的 best / holdout / stage2 real 是否被明显拖慢

验收标准是：repair 占比下降，但系统不能进入“创新很多、整体完全不收敛”的状态。

---

### 2.7 Module G: mechanism champion 反馈主搜索（闭环 3）

这是当前最难但最关键的一步。

#### 具体改动文件

**`helpers.py`**
1. 明确 mechanism champion 的元信息：
   - mechanism score
   - mechanism schema
   - mechanism cluster
   - mechanism distance from CMS
   - champion lineage tag
2. 提供 targeted seed refresh helper：
   - 用当前缺失机制轴生成 targeted seed。

**`evolution.py`**
1. `mechanism_best` 不再只记录：
   - 进入 migration extra migrant
   - 进入 elite 候选池
   - 进入 seed refresh 候选源
   - 影响 novelty target selection
   - 影响 candidate ranking bonus
2. 针对 mechanism champion 建立反馈链日志：
   - `mechanism_feedback_migration`
   - `mechanism_feedback_elite`
   - `mechanism_feedback_seed_refresh`
   - `mechanism_feedback_targeted_novelty`
3. dual champion 的生存逻辑写清楚：
   - exploit best 保持误差优势；
   - mechanism best 保持创新动力；
   - 两者长期并存，而不是二选一。

**`llm_engine.py`**
- 在 targeted novelty proposal 中，允许直接读取“当前缺失机制轴”作为 proposal hint。

#### 推进哪条闭环

- 直接推进闭环 3。

#### 期望结果

- mechanism best 不再只是日志中好看；
- 它开始真实影响 migration / elite / seed refresh / novelty targeting；
- 主搜索逐步被机制赢家反向塑造。

#### 通过标准

- `mechanism_feedback_*` 指标持续非零；
- mechanism-best lineage 能进入并影响 exploit lineage。

#### 渐进式最小反馈链（建议 rollout 顺序）

为了避免 mechanism champion 反馈过强、直接拖偏 exploit 主线，闭环 3 的 rollout 必须分三层推进：

**Level 1：低风险反馈**
- 仅影响 targeted novelty hint
- 仅影响 seed refresh 候选源
- 不直接进入 elite

**Level 2：中风险反馈**
- 进入 migration extra migrant
- 影响 candidate ranking bonus
- 仍不直接主导全局 elite

**Level 3：高风险反馈**
- 才允许进入 elite 候选池
- 才允许对 replacement / survivor 逻辑产生硬影响

默认实施顺序必须是 Level 1 -> Level 2 -> Level 3，除非日志已经证明前一层足够稳定。

---

### 2.8 Module H: primitive / runtime 架构升级（辅助模块，不是主线）

这一部分保留，但降为**辅助层**。它只能服务于上面三条闭环，不能盖过当前搜索动力学主线。

#### 具体改动文件

**`init_dex_language.py`**
1. 作为 init-side primitive source-of-truth；
2. 明确 `(x, y, z)` 三元组输出协议；
3. `list_3 / hash_salt / hash_on_slice / select_hash` 的 grammar 与常量集合成为全工程引用基线。

**`update_language.py`**
1. 作为 update-side primitive source-of-truth；
2. `write_count / update_count / update_state / query_count / query_state / *_if` 成为 update runtime 兼容基线；
3. 其参考运行时必须尊重 init_dex 三元组里的 `z/plane`。

**`query_language.py`**
1. 作为 query-side primitive source-of-truth；
2. `query_date / cnt_rdstate / base_sel / safe_min / safe_max / sum3 / median3` 成为 query runtime 兼容基线；
3. 其参考运行时必须尊重 init_dex 三元组里的 `z/plane`。

**`common.py`**
1. 统一 primitive registry；
2. 把三份 primitive 文件接成 shared registry / shared reference paths；
3. primitive tier：
   - core
   - observation
   - advanced
4. shared constants / width clamp / `list_3` 协议规范化。

**`evaluator.py`**
1. search-time / export-time runtime 语义统一；
2. evaluator 里的 update/query 运行时必须与三个 primitive 文件保持兼容；
3. 尤其要保证 `init_dex` 输出中的第三坐标 `z` 在 search-time runtime、generated runtime 中都被一致解释；
4. 如果要新增 observation primitive（如 `cnt_min3`、`of_any3`），必须共用一套 runtime。

**`helpers.py`**
1. AST simplify / legality / effect summary 同步支持新增 primitive；
2. mechanism metadata 同步新增 primitive 产生的机制轴；
3. family / mechanism 判定允许显式引用三份 primitive 文件中的 grammar 事实。

**`llm_engine.py`**
- primitive catalogue / motif card 跟着更新；
- dual-source validation 默认以这三份 primitive 文件作为 reference source。

**`cli.py`**
- primitive reference path 默认绑定与 consistency log 输出。

#### 注意

这一模块的优先级必须服从：
- numeric-risk 前门化
- seed 去坏保好
- novelty 机制分叉
- mechanism champion feedback

也就是说：
- 可以做；
- 但不能先于主线闭环大规模展开。

#### primitive / runtime 扩展准入规则

任何新 primitive 或 runtime helper，只有同时满足下面三条时才允许加入：

1. **机制表达增益**：它确实引入了当前 primitive set 无法表达的新机制轴；
2. **语义一致性**：`evaluator runtime`、`exported runtime`、AST legality / simplify 能同步接住；
3. **metadata 可感知**：新增 primitive 产生的机制轴能被 helpers / novelty / mechanism metadata 识别。

如果一个 primitive 只让表达式更花，但不增加新的机制轴，不作为当前优先级任务。

#### 防止过度扩张的约束

在闭环 1 和闭环 2 还没有明显改善之前：

- 不进行大规模 primitive tier 扩张；
- 不重写 LLM proposal grammar 主体；
- 不把 query 小修补重新抬成主创新方向。

primitive / runtime 升级只能服务主线闭环，不能把 proposal 重心带回“语法更复杂但动力学没变强”的方向。

---

## 3. Challenges & Solutions

### Challenge A: proposal 重心容易被 primitive 架构带偏

**问题：** primitive / query / geometry 改起来很显眼，但当前最堵的其实是搜索动力学闭环。

**方案：**
- 所有 primitive 升级一律降级为辅助模块；
- 任何 primitive 改动都必须先回答：它在推进哪条闭环？
- 如果与三条闭环无关，不作为当前高优先级任务。

### Challenge B: split-file 后职责边界不清，容易重复改逻辑

**问题：** 从单文件拆出后，numeric-risk、AST、novelty metadata 容易在 `evaluator.py` 和 `helpers.py` 之间重复；加入三个 primitive 文件后，grammar source-of-truth 也可能在 `common.py` / `evaluator.py` / primitive modules 之间失配。

**方案：**
- `evaluator.py` 只负责执行语义与 AST/runtime；
- `helpers.py` 只负责策略逻辑（risk / seed / novelty / mechanism metadata）；
- `evolution.py` 只负责动力学与预算分配；
- `llm_engine.py` 只负责 proposal 生成与 schema materialization；
- `init_dex_language.py / update_language.py / query_language.py` 只负责 primitive grammar 与其局部参考运行时，不重复承担搜索策略逻辑。

### Challenge C: mechanism champion 反馈后，可能破坏 exploit stability

**问题：** 若反馈过强，主搜索可能被不成熟新机制拖偏。

**方案：**
- mechanism champion feedback 先从 migration / seed refresh 小比例注入开始；
- elite feedback 最后打开；
- dual champion 长期并存，不搞 winner-take-all。

### Challenge D: repair 降权后，系统短期可能变慢

**问题：** repair 现在是稳定推进器，降权后短期 exploit 改善可能变慢。

**方案：**
- repair 不删除，只降配额；
- novelty / mechanism winner 配额渐进上调；
- 以 holdout / stage2 real 和多峰结构指标作为主要验收，而不是只看短期 best error。

---

## 4. Expected Contributions

如果这版 split-file proposal 成功落地，预期贡献不是“又整理了一次代码结构”，而是：

1. **把当前分文件工程变成真正可持续迭代的机制发现平台**，而不只是从单文件拆出来的代码壳。
2. **让三条闭环第一次在 split-file 架构中被明确、可观测、可调试地推进**：
   - numeric-risk 前门化；
   - novelty 机制多峰分叉；
   - mechanism champion 反馈主搜索。
3. **让 proposal、runtime、LLM、novelty、mechanism feedback 全部按文件职责落地**，最终输出完整可运行代码，而不是停留在概念 patch。
4. **让 split-file 工程的最终目标始终保持为“主动发现并晋升新 sketch 机制”**，而不是重新滑回 repair/CMS path。

---

## 5. Experiment Protocol

为了让每一轮 split-file 修改都能被稳定对比，实验协议必须固定为两层：

### 5.1 Quick Run（快速验证）

目标：验证闭环是否被接通，而不是追求最终最好分数。

建议：
- 120 代
- 固定 seed
- 必须输出完整日志
- 优先检查：
  - LLM 主链是否畅通
  - numeric-risk verdict 是否一致
  - seed bank / inject 是否健康
  - novelty_incubated / promoted 是否非零

### 5.2 Formal Run（正式实验）

目标：验证主搜索动力学是否真的改善。

建议：
- 300 代作为标准正式长度
- 必须固定对照 seed 跑 before / after
- 必须同时汇报：
  - best / holdout / stage2 real
  - `novelty_incubated / novelty_promoted`
  - `novelty_mechanism_cluster_count`
  - `risk_reject_by_phase / risk_reason_topk`
  - `repair_injected` vs `novelty_injected`
  - `mechanism_feedback_*`（如果闭环 3 已开启）

### 5.3 结果解释原则

任何一次实验都不能只汇报 best error。必须解释：

1. 三条闭环分别是否有推进；
2. 当前提升是 exploit 改善，还是 innovation path 真增强；
3. 是否出现了“短期分数更好，但系统重新滑回 repair/CMS path”的风险。

---

## 6. Final Delivery Standard

最终交付时，必须满足下面四条：

1. **代码层面**
   - 给出完整可运行的分文件版本代码；
   - 所有改动落在：
     - `mutate_cmsketch_refactored.py`
     - `common.py`
     - `evaluator.py`
     - `helpers.py`
     - `llm_engine.py`
     - `evolution.py`
     - `cli.py`
     - `init_dex_language.py`
     - `update_language.py`
     - `query_language.py`

2. **实验层面**
   - 给出 quick run 与 formal run 两套完整命令；
   - 区分：快速验证哪几个闭环、正式实验看哪些指标。

3. **日志层面**
   - 必须能看到三条闭环各自的 telemetry；
   - 不能只给 best error。

4. **解释层面**
   - 每一项改动都必须回答：
     - 它属于哪个阶段？
     - 它推进哪条闭环？
     - 它是在强化 innovation path，还是又把系统拉回 repair/CMS path？

---

## 7. Immediate Priority Queue

如果现在立刻开始按 split-file 工程改代码，优先级必须是：

**Priority 0**  
`common.py + cli.py + init_dex_language.py + update_language.py + query_language.py`：先把 primitive registry、reference paths、三份 primitive source-of-truth 接通，并消除 flat/package 兼容问题。

**Priority 1**  
`helpers.py + evolution.py`：继续收紧 seed 去坏保好逻辑。

**Priority 2**  
`helpers.py + evolution.py`：继续把 novelty 的判断单位往 mechanism-cluster 推。

**Priority 3**  
`helpers.py + evolution.py + llm_engine.py`：继续降低 repair 的默认主导性。

**Priority 4**  
`evolution.py + helpers.py + llm_engine.py`：尝试把 mechanism champion 接入主种群反馈。

**Priority 5**  
`common.py + evaluator.py + init_dex_language.py + update_language.py + query_language.py`：再做 primitive/runtime 辅助升级，前提是不能干扰前四项。

---

## 8. One-Sentence Summary

这份 split-file version proposal 的最高目标是：

> 以 `sketch方案new.txt` 的三条闭环和阶段顺序为主骨架，在当前分文件工程上直接施工，最终产出一套完整可运行、并真正朝“机制发现与晋升系统”推进的 sketch 搜索代码。
