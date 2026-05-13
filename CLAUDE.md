# CLAUDE.md — AIQ (NVIDIA AIRA) / RCA Agent

## 项目概述

基于 NVIDIA NeMo Agent Toolkit (nat) + LangGraph 构建的多阶段深度研究系统。原始设计为 3 阶段 Web 研究 + 报告生成流水线，现已适配 RCA（Root Cause Analysis）评测场景。

**RCA 评测模式**：保留 AIRA 原始多阶段流水线架构，但**中间状态**从 AIRA 原版的 `running_summary` (text) 改为 `causal_graph` (JSON, 与最终输出 schema 一致)。将 RAG/Tavily 搜索替换为 DuckDB parquet 工具，通过 `agent_runner.py` 接入 RolloutRunner 统一评测管道。

**关键重构**（2026-04-14）：
- 中间状态改为 CausalGraph JSON，通过共享的 `compress_*` prompts 在每个阶段构建/精炼
- 5 阶段 pipeline：`generate_queries → data_research → build_graph → reflect_on_graph → finalize_summary`
- reflect 阶段锦上添花：refine sub-loop **复用 main loop 的 sys/user/schema/tools**（不冷启动），在当前 CausalGraph 基础上做 STRENGTHEN-not-overturn 的补漏
- finalize 阶段 0 LLM 调用（透传 state 里的 graph）
- force-summary fallback 改为程序化序列化（不调 LLM，不丢信息）
- `rca_tools.py` 和 `Deep_Research/src/rca_tools.py` 字节级对齐（含 qwen JSON-string arg 兼容修复）
- 单样本 LLM 调用从老版 ~150 降到 ~24（典型）、~94（最坏）

---

## 两种运行模式

### 1. 原始模式：Deep Research 服务（nat serve）

3 阶段流水线，通过 REST API 提供服务：

```
Stage 1: generate_query    → 生成研究查询列表
Stage 2: generate_summary  → Web 研究 + 报告撰写 + 反思 + 终稿
Stage 3: artifact_qa       → 基于报告的 Q&A
```

入口：`uv run nat serve --config_file configs/config.yml --host 0.0.0.0 --port 3838`

### 2. RCA 评测模式（agent_runner.py）

保留 AIRA 多阶段流水线拓扑，但**中间状态用 CausalGraph JSON 替代 running_summary**：

```
START → generate_queries → data_research → build_graph → reflect_on_graph → finalize_summary → END
```

入口：`echo '{"question":...}' | .venv/bin/python agent_runner.py [--log-file path.log]`

---

## RCA 评测图结构（AIRA 多阶段 + CausalGraph 中间态）

```
START
  └─► generate_queries        # 对应 AIRA Stage 1: generate_query
        │  事件描述 → LLM → 1 个调查子查询（number_of_queries=1）
        │
        └─► data_research      # 替代 AIRA Stage 2: web_research
              │  1 次 run_data_exploration sub-loop（max_rounds=60）
              │  工具：think_tool, list_tables_in_directory, get_schema, query_parquet_files
              │  返回 findings (plain text) + schema_messages（供 reflect 复用）
              │
              └─► build_graph    # ★ 新（替代 summarize_sources）
                    │  findings → compress_to_graph(compress_sp + compress_up)
                    │  → causal_graph v0 (dict with nodes/edges/root_causes)
                    │
                    └─► reflect_on_graph   # ★ 新（替代 reflect_on_summary）
                          │  for i in range(num_reflections=2):
                          │    ① run_refine_exploration(sub-loop, max_rounds=15)
                          │       复用 main loop 的 sys/user/schema/tools
                          │       + 当前 graph + STRENGTHEN-not-overturn 指令
                          │    ② compress_to_graph(累积 findings) → graph v(i+1)
                          │
                          └─► finalize_summary   # 透传（0 LLM）
                                │  json.dumps(causal_graph) → final_report
                                └─► END
```

### AIRA 节点映射（RCA 重构版）

| AIRA 原始节点 | RCA 适配节点（2026-04-14 后）| 关键变化 |
|--------------|-----------------------|---------|
| `generate_query` (Stage 1) | `generate_queries` | prompt 改为 RCA 调查查询；num_queries=1 |
| `web_research` (Stage 2) | `data_research` | RAG/Tavily → parquet tool-calling 循环（max=60）；抽取 schema_messages 供 reflect |
| `summarize_sources` | **`build_graph`**（改名）| 不再生成 text summary，改用 stdin 的 `compress_*` prompts 直接生成 CausalGraph dict |
| `reflect_on_summary` | **`reflect_on_graph`**（改名）| 每轮 sub-loop 复用 main loop 上下文 + 当前 graph + STRENGTHEN 约束；完成后 compress 累积 findings 得新 graph |
| `finalize_summary` | `finalize_summary` | **0 LLM**，透传 `state["causal_graph"]` 为 final_report |

### data_research 内部工具调用循环

`run_data_exploration` 运行一个 tool-calling sub-loop：

```
SystemMessage(system_prompt) + HumanMessage(query + data_dir + "Start by calling list_tables_in_directory")
  └─► LLM.invoke (bind_tools) → AIMessage
        ├─► 有 tool_calls → 执行工具 → ToolMessage → 回到 LLM（默认 max_rounds=60）
        └─► 无 tool_calls → 返回 findings
```

**SystemMessage 只用 stdin 传入的 `system_prompt`（RCA_ANALYSIS_SP），不再叠加 DATA_RESEARCH_SP**（已删除，避免和共享 prompt 内容重复）。

**Force-summary → 结构化序列化**：循环因 max_rounds 耗尽退出且最后一轮还在 tool_calls 时，不再调 LLM 做 plain-text summary（会丢信息），而是调用 `serialize_messages_as_findings(all_msgs)` 程序化把 `tool_calls + tool_results` 按时间序列化成结构化文字（保留原始 SQL + 原始结果），直接作为 findings 返回。

### reflect_on_graph 内部细节（锦上添花）

每次 `run_refine_exploration` sub-loop 的 messages 构造：

```python
messages = [
    SystemMessage(system_prompt),                        # ① 复用 main loop
    HumanMessage("Investigation query: ..."),            # ② 复用 main loop 的 user prompt
    *schema_messages,                                    # ③ main loop 真实的
                                                         #    list_tables + get_schema
                                                         #    AIMessage + ToolMessage 配对
    HumanMessage(f"REFINE (STRENGTHEN not overturn): ...{graph_json}..."),  # ④ 当前 graph + refine 指令
]
```

好处：
- LLM 看到"我已经发现过 schema"→ 不冷启动重新发现（省 2-3 轮）
- LLM 看到当前 graph → 知道要在它基础上补漏，不是从零做调查
- STRENGTHEN 约束 → 不颠覆已有结论，只补弱点

### schema_messages 抽取

`data_research` 节点结束时调用 `extract_schema_messages(all_msgs)`，从 main loop 的工具调用历史里抽出**所有** `list_tables_in_directory` / `get_schema` 的 `AIMessage(tool_calls=...)` + `ToolMessage(result)` 配对，存到 `state["schema_messages"]`。

---

## Tools 定义

### RCA 工具集（agent_runner.py 中使用）

| Tool | 来源 | 功能 |
|------|------|------|
| `think_tool` | agent_runner.py 内联 | 反思占位工具（返回记录的思考内容） |
| `list_tables_in_directory` | rca_tools.py | 列出目录中所有 parquet 文件及元数据 |
| `get_schema` | rca_tools.py | 获取单个/多个 parquet 文件的 schema |
| `query_parquet_files` | rca_tools.py | 用 DuckDB SQL 查询 parquet 数据，token 限制 5000 |

> 不使用 tavily_search，与 RCA_ANALYSIS_SP 描述一致。

---

## Prompt 体系（2026-04-14 重构后）

**保留的 prompt 常量**（agent_runner.py 里）：

| Prompt 常量 | 对应 AIRA prompt | 用途 |
|-----|-----|-----|
| `RCA_QUERY_WRITER` | `query_writer_instructions` | 仅用于 `generate_queries` 节点；aiq-unique，其他 agent 没有此步 |

**已删除的 prompt 常量**（和共享 prompt 重叠或不再需要）：

| 删除项 | 原作用 | 为什么删 |
|-----|-----|-----|
| `DATA_RESEARCH_SP` | data_research sub-loop 的系统提示补充 | 整段和 RCA_ANALYSIS_SP 的 `<Available Tools>` / `<Analysis Instructions>` 重复 |
| `RCA_SUMMARIZER` | summarize_sources 的 text summary prompt | 中间状态改 CausalGraph 后由 compress_* 替代 |
| `RCA_REPORT_EXTENDER` | reflect 的 text extender prompt | 同上 |
| `RCA_REFLECTION` | reflect 的 knowledge-gap prompt | refine 指令改为内联在 HumanMessage 里，不再需要独立 prompt |

**共享 prompt（stdin 传入，与其他 agent 完全一致）**：

| stdin 字段 | 用在哪里 | 其他 agent 是否用 |
|-----|-----|-----|
| `system_prompt` = `RCA_ANALYSIS_SP` | `data_research` / `reflect_on_graph` 的 sub-loop SystemMessage | ✅ 所有 agent 共享 |
| `compress_system_prompt` = `COMPRESS_FINDINGS_SP` | `build_graph` + 每次 reflect 的 compress | ✅ 所有 agent 共享 |
| `compress_user_prompt` = `COMPRESS_FINDINGS_UP` | 同上 | ✅ 所有 agent 共享 |

**aiq-unique 的运行时 prompt（内联在代码，非常量）**：

| 位置 | 内容 | 为什么是 aiq-unique |
|-----|-----|-----|
| `run_refine_exploration` 最后一条 HumanMessage | "REFINE (STRENGTHEN not overturn) this root cause graph: {graph}. Pick the SINGLE weakest aspect..." | 只有 aiq 有 "在现有 graph 上锦上添花" 的 reflect 概念 |
| `run_data_exploration` HumanMessage | "Investigation query: ... Start by calling list_tables_in_directory..." | 所有 ReAct-style agent 都有类似 framing |

---

## 新增 helper 函数（2026-04-14）

| Helper | 作用 |
|---|---|
| `extract_schema_messages(all_msgs)` | 从 main loop 抽 `list_tables` + `get_schema` 的 AIMessage+ToolMessage 配对 |
| `compress_to_graph(findings_list, compress_sp, compress_up)` | 调 LLM 把累积 findings 压成 CausalGraph dict；3 次重试兜底 |
| `run_refine_exploration(...)` | refine 阶段的 sub-loop，复用 main loop 上下文 + 当前 graph + refine 指令 |
| `serialize_messages_as_findings(all_msgs)` | 把 tool_calls + tool_results 按时间序列化成结构化文字（替代 force-summary LLM 调用） |

---

## RCAState schema (TypedDict)

| 字段 | 类型 | 作用 |
|---|---|---|
| `queries` | list[dict] | 来自 generate_queries 的调查 query |
| `data_research_results` | list[str] | 主 data_research 返回的 findings 文本 |
| `schema_messages` | list | main loop 抽出的 schema 发现 AIMessage+ToolMessage 配对 |
| `causal_graph` | dict | 当前 CausalGraph（v0/v1/v2），schema 与最终输出一致 |
| `accumulated_findings` | list[str] | main + 各轮 refine 累积的全部 findings，compress 时全量传入 |
| `final_report` | str | `json.dumps(causal_graph)` 的结果 |
| `all_tool_messages` | list (Annotated, operator.add) | 跨节点累积的完整工具调用轨迹 |

---

## 环境管理

**工具：`uv`**（见 `pyproject.toml`）

```bash
# 安装依赖
uv sync

# 额外安装 RCA 评测所需的 duckdb
uv pip install duckdb

# 运行 agent_runner
echo '...' | uv run python agent_runner.py
```

Python 要求：`>=3.12`

主要依赖：`langgraph`, `langchain-openai`, `langchain-core`, `duckdb`, `python-dotenv`

---

## 环境变量

在项目根目录创建 `.env` 文件：

```
# Qwen3.5-plus via Aliyun Coding Plan
OPENAI_API_KEY=                                          # 运行前命令行传入
OPENAI_BASE_URL=https://coding.dashscope.aliyuncs.com/v1
```

> 模型通过 `RCA_MODEL` 环境变量传入（RolloutRunner 从 YAML config 的 `model_name` 字段注入）。
> `model_factory.py`（从 Deep_Research 复制）统一处理：Claude → ChatAnthropic，其余 → ChatOpenAI。

---

## RCA 评测 stdin/stdout 接口

**stdin (6 字段):**
```json
{
  "question": "augmented_question 原文",
  "system_prompt": "RCA_ANALYSIS_SP (已 format date)",
  "user_prompt": "RCA_ANALYSIS_UP (已 format incident_description)",
  "compress_system_prompt": "COMPRESS_FINDINGS_SP",
  "compress_user_prompt": "COMPRESS_FINDINGS_UP",
  "data_dir": "/path/to/eval-data/<exp_id>/data_XXXXXXXX"
}
```

**stdout (最后一行 JSON):**
```json
{
  "output": "{\"nodes\": [...], \"edges\": [...], \"root_causes\": [...]}",
  "trajectory": [
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "...", "tool_call_id": "..."}
  ]
}
```

**关键要求：**
- `output` 必须是纯 JSON，不能有 markdown 代码块包裹
- trajectory 必须是 OpenAI role 格式（影响 tool_bonus 计算）
- 成功必须 exit 0

---

## 数据流（RCA 评测模式）

```
stdin JSON
  │
  ▼
agent_runner.py::main()
  │  解析 payload → 构建 config["configurable"]
  │
  ▼
build_agent()
  │  构建 AIRA 多阶段 StateGraph:
  │  generate_queries → data_research → summarize_sources
  │    → reflect_on_summary → finalize_summary
  │
  ▼
agent.invoke(initial_state, config)
  │  Stage 1: 事件描述 → 1 个调查子查询 (1 LLM)
  │  Stage 2: 主 tool-loop (max=60) → findings + schema_messages
  │  Stage 3: compress(findings) → graph_v0 (1 LLM)
  │  Stage 4: for i in range(num_reflections=2):
  │             refine sub-loop (max=15, 复用上下文 + graph) → new findings
  │             compress(累积 findings) → graph_v(i+1) (1 LLM per iteration)
  │  Stage 5: finalize (0 LLM, 透传 graph)
  │
  ▼
stdout: {"output": json.dumps(causal_graph), "trajectory": [...], "usage": {...}}

典型 LLM 调用数:   1 + 8 + 1 + 2×(6+1) + 0 = ~24
最坏 LLM 调用数: 1 + 60 + 1 + 2×(15+1) + 0 = ~94
```

---

## 原始项目结构

```
aiq/
├── agent_runner.py              # RCA 评测接口（新增）
├── model_factory.py             # 统一模型工厂（从 Deep_Research 复制）
├── rca_tools.py                 # DuckDB parquet 工具（新增，来自 Deep_Research）
├── CLAUDE.md                    # 本文档
├── pyproject.toml               # 依赖声明（uv 管理）
├── .env                         # API 密钥（百炼 Coding Plan endpoint）
│
├── aira/src/aiq_aira/           # 原始 AIRA 包
│   ├── functions/
│   │   ├── generate_queries.py  # Stage 1: 查询生成图
│   │   ├── generate_summary.py  # Stage 2: 研究 + 报告图
│   │   └── artifact_qa.py       # Stage 3: Q&A 图
│   ├── nodes.py                 # LangGraph 节点实现
│   ├── schema.py                # Pydantic 模型 + AIRAState
│   ├── prompts.py               # 所有 prompt 模板
│   ├── tools.py                 # RAG + Tavily 搜索工具
│   ├── search_utils.py          # 搜索辅助函数
│   ├── report_gen_utils.py      # 报告生成辅助
│   ├── register.py              # nat 插件注册
│   └── constants.py             # 常量
│
├── configs/
│   ├── config.yml               # nat 服务配置
│   └── security_config.yml      # 安全模式
│
└── data/                        # 样例数据
```

---

## 关键约束

| 约束 | 值 |
|------|---|
| LangGraph 递归限制 | 100 |
| main `data_research` tool-loop max_rounds | **60**（自然收敛 ~8 轮典型） |
| `reflect_on_graph` 内部 `run_refine_exploration` max_rounds | **15** |
| `num_reflections` (reflect iterations) | **2**（串行，后一轮基于前一轮的 graph） |
| `number_of_queries` (generate_queries 生成的 query 数) | **1** |
| DuckDB 结果 token 限制 | 5000 tokens |
| Python 版本 | >=3.12 |
| 模型（RCA 评测） | qwen3.5-plus（百炼 Coding Plan） |

---

## --log-file 参数

`agent_runner.py` 支持 `--log-file <path>` 参数：

```bash
.venv/bin/python agent_runner.py --log-file logs/aiq_idx_5.log < payload.json
```

把 `INFO` 级别的 logging 输出同时写到文件（和 stderr 并存），用于 run_rollout_with_retry.py 批量跑时实时查看每个样本的推理过程。在 `RolloutRunner/scripts/run_rollout_with_retry.py` 里通过 `--log_dir` 参数为每个样本生成独立日志文件（`idx_N_sample_M.log`）。

---

## rca_tools.py 同步

`aiq/rca_tools.py` 和 `Deep_Research/src/rca_tools.py` **字节级对齐**，含 qwen JSON-string arg 兼容修复：

```python
# Some models (e.g. Qwen) serialize list args as JSON strings:
#   '["file1.parquet", "file2.parquet"]' or '"file.parquet"'
if stripped.startswith("["):
    try:
        parquet_files = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        parquet_files = [parquet_files]
elif stripped.startswith('"') and stripped.endswith('"'):
    parquet_files = [stripped.strip('"')]
```

**维护原则**：`aiq/rca_tools.py` 不应独立修改，任何修复都应在 `Deep_Research/src/rca_tools.py` 做然后 sync 过来（`diff` 应永远为 0），避免 aiq/thinkdepthai 在工具层面产生行为差异。

---

## 常用命令

```bash
# === RCA 评测 ===
# 冒烟测试
cd /home/nn/SOTA-agents/RolloutRunner
python scripts/run_rollout.py --agent aiq --source_exp_id <exp_id> --limit 1

# 全量运行
nohup python -u scripts/run_rollout.py --agent aiq --source_exp_id <exp_id> \
  > rollout_aiq.log 2>&1 &

# === 原始服务模式 ===
cd /home/nn/SOTA-agents/aiq
uv run nat serve --config_file configs/config.yml --host 0.0.0.0 --port 3838
```
