# Patch Summary

本次补丁完成了三类工作：

1. **proposal 补全**
   - 把 `init_dex_language.py` / `update_language.py` / `query_language.py` 纳入 proposal 的直接施工对象。
   - 更新了 Module A / Module B / Module H / Final Delivery / Priority Queue。
   - 明确三份 primitive-set 文件是 grammar/runtime/reference 的 source-of-truth。

2. **primitive-set 兼容性修复**
   - `update_language.py` 修复了 `overflow_state` 变量名错误，统一为 `overflow_matrices`。
   - `update_language.py` / `query_language.py` 新增 `_resolve_loc(...)`，并开始尊重 `init_dex` 产生的 `(x, y, z)` 三元组。
   - `init_dex_language.py` 规范化了 `list_3(...)` 的输出协议与 mod 处理。

3. **split-file 工程兼容性修复**
   - `common.py` 新增 primitive registry、reference-path 自动发现、primitive consistency report。
   - `cli.py` 为空时自动注入 `--llm_ref_*_pset_path`，并输出 primitive registry 日志。
   - `evaluator.py` 的 search-time runtime 修复为与 primitive-set 三元组语义一致。
   - `evaluator.py` 导出的完整脚本对 `init_dex_language` 增加了 flat/package 双模式 import 回退。
   - `cli.py` / `evaluator.py` / `helpers.py` / `llm_engine.py` / `evolution.py` / `mutate_cmsketch_refactored.py` 都增加了 package/flat 双模式 import 回退。
