#!/usr/bin/env python
"""
agent_runner.py — AIQ (NVIDIA AIRA) RCA 测评接口

保留 AIRA 原始多阶段 LangGraph 流水线架构，但中间状态从 running_summary (text)
改为 causal_graph (JSON, 与最终输出 schema 一致)：

  generate_queries   → 将事件描述分解为调查子查询
  data_research      → 主 sub-loop（max=60）用 parquet 工具探索 + 抽 schema 消息
  build_graph        → 复用 stdin compress_* 把 findings 压成 graph_v0
  reflect_on_graph   → 串行 num_reflections 轮 refine sub-loop（max=10）：
                       每轮复用 main loop 的 sys/user/schema/tools + 当前 graph
                       + STRENGTHEN 指令，跑完 compress 累积 findings 得新 graph
  finalize_summary   → 透传 graph 为最终 CausalGraph JSON（0 LLM）

工具替换：RAG/Tavily Web Search → DuckDB Parquet 工具（与 thinkdepthai 相同）
模型：通过 RCA_MODEL 环境变量传入
接口：RolloutRunner stdin/stdout 标准接口

stdin:  JSON { question, system_prompt, user_prompt,
               compress_system_prompt, compress_user_prompt, data_dir }
stdout: JSON { output (CausalGraph JSON), trajectory (OpenAI 格式), usage }
"""
import argparse
import json
import logging
import operator
import os
import re
import sys
from pathlib import Path
from typing import Annotated

try:
    from sota_rca.tracker import auto_install
    _tracker = auto_install()
except ImportError:
    _tracker = None  # sota_rca not on PYTHONPATH; tracker disabled

# 根据模型选择 hook：Claude 走 Anthropic SDK，其余走 OpenAI SDK
_RCA_MODEL = os.environ.get("UTU_LLM_MODEL", os.environ.get("RCA_MODEL", "claude-sonnet-4-6"))
if _RCA_MODEL.startswith("claude"):
    _tracker.install_anthropic_hooks()




from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from model_factory import create_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig

from rca_tools import get_schema, list_tables_in_directory, query_parquet_files

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# ── 模型配置 ──────────────────────────────────────────────────────────────

# MODEL_NAME = "openai:doubao-seed-2-0-pro-260215"
# MODEL_NAME = "openai:kimi-k2-0905-preview"
# MODEL_NAME = "openai:openai/claude-sonnet-4-6"
# MODEL_NAME = "openai:gemini-3.1-pro-preview"
# MODEL_NAME = "openai:openai/claude-sonnet-4-6"
# MODEL_NAME = "openai:claude-sonnet-4-6"

RCA_MODEL = os.environ.get("UTU_LLM_MODEL", os.environ.get("RCA_MODEL", "claude-sonnet-4-6"))


def _make_model(max_tokens: int = 32768):
    """Create LLM via model_factory. Model name from RCA_MODEL env var."""
    return create_model(RCA_MODEL, max_tokens=max_tokens)


# ── think_tool（与 thinkdepthai 相同）──────────────────────────────────────

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each round of data queries to analyze results and plan next steps.

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded
    """
    return f"Reflection recorded: {reflection}"


# ── RCA 工具集 ────────────────────────────────────────────────────────────

RCA_TOOLS = [think_tool, list_tables_in_directory, get_schema, query_parquet_files]
RCA_TOOLS_BY_NAME = {t.name: t for t in RCA_TOOLS}


# ── State（保留 AIRA AIRAState 结构）──────────────────────────────────────

class RCAState(TypedDict):
    """
    对应 AIRA 的 AIRAState，适配 RCA 场景。中间状态用 causal_graph (JSON, 与最终
    输出 schema 一致) 替代了原 AIRA 的 running_summary (text)。

    字段映射 / 新增：
      queries               → 调查查询列表（来自 generate_queries）
      data_research_results → 主 data_research 的 findings 文本（list[str]）
      schema_messages       → main loop 抽出的 schema 发现消息对（list_tables +
                               get_schema 的 AIMessage + ToolMessage 配对），供 refine
                               sub-loop 复用，避免冷启动重新发现
      causal_graph          → 当前的 CausalGraph (dict)，每次 reflect 后更新
      accumulated_findings  → main loop + 各轮 reflect sub-loop 的全部 findings 文本，
                               compress 时累积传给 LLM
      final_report          → 最终输出（causal_graph 序列化后的 JSON 字符串）
      all_tool_messages     → 完整工具调用轨迹（用于 stdout trajectory）
    """
    queries: list[dict]
    data_research_results: list[str]
    schema_messages: list
    causal_graph: dict
    accumulated_findings: list[str]
    final_report: str
    all_tool_messages: Annotated[list, operator.add]  # 跨节点累积


# ── Prompts（适配自 AIRA prompts.py，改为 RCA 场景）──────────────────────

# 对应 AIRA query_writer_instructions
RCA_QUERY_WRITER = """You are the investigation-query architect for an RCA (Root Cause Analysis) agent that analyzes microservice incidents using telemetry data (logs, metrics, traces in parquet format).

Given an incident description, generate {number_of_queries} investigation queries to systematically explore the telemetry data.

# Incident Description
{incident}

# Instructions
- Design queries that cover different investigation angles:
  * Service error rates and HTTP status codes
  * Latency anomalies and response time spikes
  * Log error patterns and exception messages
  * Trace call chains and service dependencies
  * Resource utilization (CPU, memory) stress indicators
- Each query should be specific enough to guide targeted SQL queries on parquet data
- Format your response as a JSON list:

```json
[
    {{"query": "Investigate error rates across services by comparing normal vs abnormal metrics", "report_section": "Error Analysis", "rationale": "Identify which services have elevated error rates during the incident"}},
    {{"query": "Analyze trace data to find latency spikes and failing call chains", "report_section": "Trace Analysis", "rationale": "Trace the propagation path of the failure"}}
]
```"""

# 注：已删除 DATA_RESEARCH_SP / RCA_SUMMARIZER / RCA_REPORT_EXTENDER / RCA_REFLECTION
# 这四个 prompt 与 RolloutRunner 的共享 RCA_ANALYSIS_SP 内容大量重复（工具列表、调查
# 流程、根因方向等都已在 stdin 传入的 system_prompt 里）。中间状态从 running_summary
# (text) 改为 causal_graph (JSON) 后：
#   - data_research 的 sub-loop SystemMessage 只用 stdin 的 system_prompt（去重）
#   - graph 构建/更新统一通过 compress_system_prompt + compress_user_prompt
#   - reflect 的指令直接内联在 HumanMessage 里，复用 main loop 的 sys/user/schema/tools


# ── Helper: 工具调用数据探索（替代 AIRA 的 process_single_query）──────────

def run_data_exploration(
    query: str,
    data_dir: str,
    system_prompt: str = "",
    max_rounds: int = 60,
) -> tuple[str, list]:
    """
    替代 AIRA 的 process_single_query（search_utils.py）。
    用 LLM + parquet 工具进行数据探索，类似 AIRA 的 search_rag + search_tavily。

    SystemMessage 直接用 stdin 传入的 system_prompt（RCA_ANALYSIS_SP），
    不再叠加 DATA_RESEARCH_SP（已删除，避免和 RCA_ANALYSIS_SP 内部重复）。

    Args:
        system_prompt: RCA 领域系统提示（来自 RolloutRunner 的 RCA_ANALYSIS_SP）
        max_rounds:    tool-loop 硬上限。主 data_research 默认 60（接近 thinkdepthai
                       max 91 但留余量给 context window），refine 调用时传 10。
                       到上限时再强制调一次无 tools 的 LLM 让它基于已有证据收尾
                       （参考 openrca controller.py:136 的"最大步数到达"分支）。

    Returns:
        (findings_text, tool_messages_list)
    """
    model = _make_model()
    model_with_tools = model.bind_tools(RCA_TOOLS)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Investigation query: {query}\n\n"
            f"Data location: `{data_dir}`\n"
            f"Start by calling `list_tables_in_directory(directory=\"{data_dir}\")` "
            f"to discover available parquet files."
        )),
    ]

    all_msgs = []
    response = None
    for _round in range(max_rounds):
        response = model_with_tools.invoke(messages)
        messages.append(response)
        all_msgs.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            tool_fn = RCA_TOOLS_BY_NAME[tc["name"]]
            result = tool_fn.invoke(tc["args"])
            tool_msg = ToolMessage(
                content=result, name=tc["name"], tool_call_id=tc["id"]
            )
            messages.append(tool_msg)
            all_msgs.append(tool_msg)

    # 兜底：循环因 range 耗尽退出且最后一轮还在 tool_calls → 把原始调查史
    # 序列化为结构化 findings 文字（不调 LLM，不丢信息），下游 compress 直接用。
    if response is not None and response.tool_calls:
        logger.warning(
            f"run_data_exploration hit max_rounds={max_rounds}, "
            f"serializing raw tool history as findings (skipping plain-text summary)"
        )
        findings = serialize_messages_as_findings(all_msgs)
    else:
        findings = str(response.content) if response is not None and response.content else ""

    return findings, all_msgs


# ── Helper: 将 tool loop 的 all_msgs 序列化为结构化 findings 文字 ─────────

def serialize_messages_as_findings(all_msgs: list) -> str:
    """
    当 tool loop 因 max_rounds 耗尽退出时用。把 all_msgs 中的 tool_calls +
    tool_results 按时间顺序拼成结构化文字，保留每条 SQL 的原始参数和结果。

    比调 LLM 做 plain text summary 更好：
      - 无信息丢失（SQL 和 result 原样保留）
      - 省 1 次 LLM 调用
      - compress_to_graph 下游看到完整调查史，由 compress 的 LLM 自己决定取舍

    格式示例：
      ### Step 1: list_tables_in_directory
      Arguments: {"directory": "/path"}
      Result: abnormal_logs.parquet (50000 rows), ...

      ### Step 2: query_parquet_files
      Arguments: {"query": "SELECT ..."}
      Result: ts-order-service: 5000, ...
    """
    # 建立 tool_call_id → ToolMessage content 的映射
    tool_results: dict[str, str] = {}
    for m in all_msgs:
        tc_id = getattr(m, "tool_call_id", None)
        if tc_id:
            tool_results[tc_id] = str(m.content)

    parts: list[str] = []
    step_idx = 0
    for m in all_msgs:
        tool_calls = getattr(m, "tool_calls", None) or []
        if tool_calls:
            for tc in tool_calls:
                step_idx += 1
                tc_name = tc.get("name", "?")
                tc_args = tc.get("args", {})
                try:
                    args_str = json.dumps(tc_args, ensure_ascii=False)
                except Exception:
                    args_str = str(tc_args)
                parts.append(f"### Step {step_idx}: {tc_name}")
                parts.append(f"Arguments: {args_str[:1500]}")
                tc_id = tc.get("id")
                if tc_id and tc_id in tool_results:
                    parts.append(f"Result: {tool_results[tc_id][:3000]}")
                parts.append("")
        else:
            # AIMessage 纯文字（可能是 LLM 在某一轮决定不调工具时的思考）
            content = getattr(m, "content", None)
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") if isinstance(b, dict) and b.get("type") == "text" else ""
                    for b in content
                ).strip()
            if content and isinstance(m, AIMessage):
                parts.append("### Reasoning")
                parts.append(str(content)[:2000])
                parts.append("")

    return "\n".join(parts) if parts else "(no investigation steps recorded)"


# ── Helper: 抽取 schema 发现消息（供 reflect 复用，避免冷启动）──────────────

def extract_schema_messages(all_msgs: list) -> list:
    """
    从 main loop 的 all_msgs 里抽出 list_tables_in_directory / get_schema 的
    AIMessage (tool_calls) + ToolMessage (results) 配对。

    refine sub-loop 把这些消息塞进自己的 messages 列表，让 LLM 看到"我已经发现过
    schema"，省掉重新调 list_tables/get_schema 的 2-3 轮冷启动。
    """
    schema_msgs: list = []
    schema_tool_names = {"list_tables_in_directory", "get_schema"}
    i = 0
    while i < len(all_msgs):
        m = all_msgs[i]
        tool_calls = getattr(m, "tool_calls", None) or []
        relevant_ids = {tc["id"] for tc in tool_calls if tc.get("name") in schema_tool_names}
        if relevant_ids:
            schema_msgs.append(m)
            j = i + 1
            while j < len(all_msgs):
                tc_id = getattr(all_msgs[j], "tool_call_id", None)
                if tc_id in relevant_ids:
                    schema_msgs.append(all_msgs[j])
                    j += 1
                else:
                    break
            i = j
            continue
        i += 1
    return schema_msgs


# ── Helper: compress findings → CausalGraph (复用 stdin compress_*)────────

def compress_to_graph(
    accumulated_findings: list[str],
    compress_sp: str,
    compress_up: str,
    max_retries: int = 3,
) -> dict:
    """
    用 stdin 传入的 compress_system_prompt + compress_user_prompt 把累积 findings
    压成 CausalGraph dict。每次都从全部 findings 重新生成（让新证据自然修正旧
    结论），不增量更新 graph。

    失败兜底：3 次重试解析失败 → 返回空 graph，不崩溃。
    """
    findings_text = "\n\n---\n\n".join(accumulated_findings)
    llm = _make_model(max_tokens=32000)

    last_err: str | None = None
    for attempt in range(max_retries):
        messages = [
            SystemMessage(content=compress_sp),
            HumanMessage(
                content=(
                    f"Here is my complete RCA investigation findings:\n\n"
                    f"{findings_text}\n\n"
                    f"{compress_up}"
                )
            ),
        ]
        response = llm.invoke(messages)
        text = strip_markdown_json(strip_think_tags(str(response.content)))
        try:
            graph = json.loads(text)
            if isinstance(graph, dict):
                return graph
            last_err = "compress output is not a dict"
        except json.JSONDecodeError as e:
            last_err = f"JSONDecodeError: {e}"
        logger.warning(
            f"compress_to_graph attempt {attempt + 1}/{max_retries} failed: {last_err}"
        )

    logger.error(
        f"compress_to_graph failed all {max_retries} attempts; returning empty graph"
    )
    return {"nodes": [], "edges": [], "root_causes": []}


# ── Helper: refine sub-loop（复用 main loop 上下文，锦上添花当前 graph）──────

def run_refine_exploration(
    original_query: str,
    data_dir: str,
    system_prompt: str,
    current_graph: dict,
    schema_msgs: list,
    max_rounds: int = 15,
) -> tuple[str, list]:
    """
    refine 阶段的 sub-loop。除了"前期探索的 SQL/结果文字"之外，全部复用 main loop
    的上下文：
      - SystemMessage (stdin 的 system_prompt)
      - HumanMessage (与 main loop 一致的 investigation query 模板)
      - main loop 真实的 schema 发现 AIMessage + ToolMessage 配对
      - bind_tools 的工具列表
      - 当前 CausalGraph (要锦上添花的对象)
      - "STRENGTHEN not overturn" 的 refine 指令

    LLM 看到这套上下文后会自然产生"我已经查过 schema → 我之前的结论是这个 graph
    → 现在该补哪个弱点 → 跑几条 SQL → 返回发现"的连贯推理。

    Returns:
        (findings_text, tool_messages_list)
    """
    model = _make_model()
    model_with_tools = model.bind_tools(RCA_TOOLS)

    messages: list = [
        # ① 复用 main loop 的 SystemMessage
        SystemMessage(content=system_prompt),
        # ② 复用 main loop 的 HumanMessage（保持上下文一致）
        HumanMessage(content=(
            f"Investigation query: {original_query}\n\n"
            f"Data location: `{data_dir}`\n"
            f"Start by calling `list_tables_in_directory(directory=\"{data_dir}\")` "
            f"to discover available parquet files."
        )),
        # ③ 直接塞入 main loop 的真实 schema 发现消息对（AIMessage + ToolMessage）
        *schema_msgs,
        # ④ 追加 refine 指令 + 当前 CausalGraph
        HumanMessage(content=(
            f"You have already discovered the data schema above. Now you need to "
            f"REFINE (strengthen, not overturn) the preliminary root cause graph "
            f"produced from your earlier investigation:\n\n"
            f"```json\n{json.dumps(current_graph, ensure_ascii=False, indent=2)}\n```\n\n"
            f"## Your task\n\n"
            f"Pick the SINGLE weakest aspect of this graph:\n"
            f"- A suspected root cause with no/thin evidence\n"
            f"- An edge claimed as causal but only supported by correlation\n"
            f"- A service on the fault path that wasn't investigated\n"
            f"- A missing baseline comparison (normal vs abnormal)\n\n"
            f"Then gather additional SQL evidence to STRENGTHEN it. Rules:\n"
            f"- STRENGTHEN, do not overturn well-supported conclusions\n"
            f"- Use `query_parquet_files` directly; do NOT re-run "
            f"`list_tables_in_directory` or `get_schema` for tables you already know\n"
            f"- Target 5-8 tool calls. When you have your refinement evidence, "
            f"stop calling tools and return your findings as plain text."
        )),
    ]

    all_msgs: list = []
    response = None
    for _round in range(max_rounds):
        response = model_with_tools.invoke(messages)
        messages.append(response)
        all_msgs.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            tool_fn = RCA_TOOLS_BY_NAME[tc["name"]]
            result = tool_fn.invoke(tc["args"])
            tool_msg = ToolMessage(
                content=result, name=tc["name"], tool_call_id=tc["id"]
            )
            messages.append(tool_msg)
            all_msgs.append(tool_msg)

    # 兜底：同 run_data_exploration，把 refine 的原始调查史序列化为 findings 文字，
    # 不调 LLM 做 plain text summary，避免信息丢失，让下游 compress 直接处理。
    if response is not None and response.tool_calls:
        logger.warning(
            f"run_refine_exploration hit max_rounds={max_rounds}, "
            f"serializing raw refinement history as findings"
        )
        findings = serialize_messages_as_findings(all_msgs)
    else:
        findings = str(response.content) if response is not None and response.content else ""

    return findings, all_msgs


def strip_think_tags(text: str) -> str:
    """清理 <think>...</think> 标签（对应 AIRA report_gen_utils.py 的逻辑）。"""
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    while "</think>" in text:
        end = text.find("</think>") + len("</think>")
        text = text[end:]
    return text


# ── Node 1: generate_queries（对应 AIRA Stage 1: generate_query）─────────

def generate_queries(state: RCAState, config: RunnableConfig) -> dict:
    """
    对应 AIRA 的 generate_query 节点（nodes.py:generate_query）。
    从事件描述生成调查子查询列表。

    AIRA 原始流程：
      topic + report_organization → query_writer_instructions → LLM → parse JSON → GeneratedQuery[]
    RCA 适配：
      incident_description → RCA_QUERY_WRITER → LLM → parse JSON → query dicts
    """
    logger.info("GENERATE QUERIES")
    llm = _make_model()
    # 使用 augmented question 作为事件描述（含数据路径信息）
    incident = config["configurable"].get("question") or config["configurable"]["user_prompt"]
    number_of_queries = config["configurable"].get("number_of_queries", 1)

    prompt = RCA_QUERY_WRITER.format(
        incident=incident, number_of_queries=number_of_queries
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    text = strip_think_tags(str(response.content))

    # 解析 JSON 查询列表（对应 AIRA 的 parse_json_markdown + GeneratedQuery 验证）
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            queries = json.loads(m.group(0))
        except Exception:
            queries = [
                {
                    "query": incident,
                    "report_section": "Full Analysis",
                    "rationale": "Direct investigation",
                }
            ]
    else:
        queries = [
            {
                "query": incident,
                "report_section": "Full Analysis",
                "rationale": "Direct investigation",
            }
        ]

    # 硬截断：LLM 可能忽略 number_of_queries 参数，生成更多查询
    max_queries = number_of_queries
    if len(queries) > max_queries:
        logger.info(f"Truncating {len(queries)} queries to {max_queries}")
        queries = queries[:max_queries]

    logger.info(f"Generated {len(queries)} investigation queries")
    return {"queries": queries}


# ── Node 2: data_research（替代 AIRA Stage 2: web_research）──────────────

def data_research(state: RCAState, config: RunnableConfig) -> dict:
    """
    主数据探索节点。对每个查询跑一次 run_data_exploration sub-loop（max=60）。
    findings 累积到 accumulated_findings，schema 发现消息抽出存到 schema_messages
    供后续 reflect 复用（避免冷启动重复 list_tables/get_schema）。
    """
    logger.info("STARTING DATA RESEARCH")
    data_dir = config["configurable"]["data_dir"]
    queries = state.get("queries") or []
    system_prompt = config["configurable"].get("system_prompt", "")

    all_findings: list[str] = []
    all_msgs: list = []

    for q in queries:
        query_text = q["query"] if isinstance(q, dict) else str(q)
        logger.info(f"Researching: {query_text[:80]}...")
        findings, msgs = run_data_exploration(
            query_text, data_dir, system_prompt, max_rounds=60
        )
        all_findings.append(findings)
        all_msgs.extend(msgs)

    schema_msgs = extract_schema_messages(all_msgs)
    logger.info(
        f"Data research complete: {len(all_findings)} findings, "
        f"{len(schema_msgs)} schema messages saved for reflect"
    )

    return {
        "data_research_results": all_findings,       # plain text findings
        "schema_messages": schema_msgs,              # for refine sub-loop reuse
        "accumulated_findings": list(all_findings),  # 累积 findings 给 compress
        "all_tool_messages": all_msgs,
    }


# ── Node 3: build_graph（compress 主 findings 为 CausalGraph v0）─────────

def build_graph(state: RCAState, config: RunnableConfig) -> dict:
    """
    用 stdin 的 compress_* prompts 把 main loop 的 findings 压成 CausalGraph v0。
    这是中间状态的初始版本，后续 reflect 会在它基础上锦上添花。
    """
    logger.info("BUILD GRAPH v0")
    compress_sp = config["configurable"]["compress_system_prompt"]
    compress_up = config["configurable"]["compress_user_prompt"]
    accumulated = state.get("accumulated_findings") or []

    graph = compress_to_graph(accumulated, compress_sp, compress_up)
    logger.info(
        f"graph_v0 built: {len(graph.get('nodes', []))} nodes, "
        f"{len(graph.get('edges', []))} edges, "
        f"{len(graph.get('root_causes', []))} root_causes"
    )
    return {"causal_graph": graph}


# ── Node 4: reflect_on_graph（refine sub-loop × num_reflections）─────────

def reflect_on_graph(state: RCAState, config: RunnableConfig) -> dict:
    """
    串行 refine：每一轮在当前 graph 上锦上添花。

    每轮执行：
      1. run_refine_exploration: 复用 main loop 的 sys/user/schema/tools，
         加上 current_graph + STRENGTHEN refine 指令，sub-loop 跑 max=10 轮
      2. compress_to_graph: 把累积 findings 重新压成新 graph

    每轮的 sub-loop 看到的 graph 是上一轮 refine 后的最新版本，
    所以是真正在前一版的基础上"锦上添花"。
    """
    logger.info("REFLECT (graph-based refine)")
    data_dir = config["configurable"]["data_dir"]
    num_reflections = config["configurable"].get("num_reflections", 2)
    system_prompt = config["configurable"].get("system_prompt", "")
    compress_sp = config["configurable"]["compress_system_prompt"]
    compress_up = config["configurable"]["compress_user_prompt"]
    original_query = (
        config["configurable"].get("question")
        or config["configurable"].get("user_prompt", "")
    )

    graph = state.get("causal_graph") or {"nodes": [], "edges": [], "root_causes": []}
    schema_msgs = state.get("schema_messages") or []
    accumulated = list(state.get("accumulated_findings") or [])
    all_msgs: list = []

    if not schema_msgs:
        logger.warning(
            "reflect_on_graph: schema_messages is empty, "
            "refine sub-loop will run without prior schema context"
        )

    for i in range(num_reflections):
        logger.info(f"Refine iteration {i + 1}/{num_reflections}")

        findings, msgs = run_refine_exploration(
            original_query=original_query,
            data_dir=data_dir,
            system_prompt=system_prompt,
            current_graph=graph,
            schema_msgs=schema_msgs,
            max_rounds=10,
        )
        all_msgs.extend(msgs)

        if not findings or not findings.strip():
            logger.warning(
                f"Refine iteration {i + 1} produced empty findings, "
                f"keeping previous graph"
            )
            continue

        accumulated.append(findings)

        new_graph = compress_to_graph(accumulated, compress_sp, compress_up)
        # 防御：如果 compress 失败返回了空 graph，保留上一版本不退化
        if new_graph.get("nodes") or new_graph.get("root_causes"):
            graph = new_graph
            logger.info(
                f"graph_v{i + 1}: {len(graph.get('nodes', []))} nodes, "
                f"{len(graph.get('edges', []))} edges, "
                f"{len(graph.get('root_causes', []))} root_causes"
            )
        else:
            logger.warning(
                f"Refine iteration {i + 1}: compress returned empty graph, "
                f"keeping previous version"
            )

    logger.info("Reflection complete")
    return {
        "causal_graph": graph,
        "accumulated_findings": accumulated,
        "all_tool_messages": all_msgs,
    }


# ── Node 5: finalize_summary（透传 graph，0 LLM 调用）─────────────────────

def finalize_summary(state: RCAState, config: RunnableConfig) -> dict:
    """
    最终阶段：直接把 causal_graph 序列化成 JSON 字符串作为 final_report。
    不再调 LLM（compress 已经在 build_graph 和 reflect_on_graph 里做过了）。
    """
    logger.info("FINALIZING REPORT (passthrough)")
    graph = state.get("causal_graph") or {"nodes": [], "edges": [], "root_causes": []}
    return {"final_report": json.dumps(graph, ensure_ascii=False)}


# ── Build Graph（保留 AIRA 的多阶段流水线拓扑）────────────────────────────

def build_agent():
    """
    保留 AIRA 的多阶段流水线拓扑。

    AIRA 原始图：
      Stage 1 (generate_queries): START → generate_query → END
      Stage 2 (generate_summary): START → web_research → summarize_sources
                                        → reflect_on_summary → finalize_summary → END

    RCA 适配（中间状态从 text 改为 CausalGraph JSON）：
      START → generate_queries → data_research → build_graph
            → reflect_on_graph → finalize_summary → END
    """
    builder = StateGraph(RCAState)

    builder.add_node("generate_queries", generate_queries)
    builder.add_node("data_research", data_research)         # 替代 web_research
    builder.add_node("build_graph", build_graph)             # 替代 summarize_sources
    builder.add_node("reflect_on_graph", reflect_on_graph)   # 替代 reflect_on_summary
    builder.add_node("finalize_summary", finalize_summary)

    builder.add_edge(START, "generate_queries")
    builder.add_edge("generate_queries", "data_research")
    builder.add_edge("data_research", "build_graph")
    builder.add_edge("build_graph", "reflect_on_graph")
    builder.add_edge("reflect_on_graph", "finalize_summary")
    builder.add_edge("finalize_summary", END)

    return builder.compile()


# ── 工具函数 ─────────────────────────────────────────────────────────────

def strip_markdown_json(text: str) -> str:
    """剥离 LLM 返回的 ```json ... ``` 代码块，提取纯 JSON。"""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


# ── LangChain → OpenAI 格式转换（与 thinkdepthai 相同）────────────────────

def to_openai_message(msg) -> dict | None:
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": str(msg.content)}

    if isinstance(msg, AIMessage):
        tool_calls = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"], ensure_ascii=False),
                },
            }
            for tc in (msg.tool_calls or [])
        ]
        # ChatAnthropic returns content as list of blocks; extract text parts
        content = msg.content
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) and b.get("type") == "text" else ""
                for b in content
            ).strip()
        entry: dict = {
            "role": "assistant",
            "content": str(content) if content else "",
        }
        if tool_calls:
            entry["tool_calls"] = tool_calls
        return entry

    if isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "content": str(msg.content),
            "tool_call_id": msg.tool_call_id,
        }

    return None


def convert_trajectory(messages: list) -> list[dict]:
    return [m for msg in messages if (m := to_openai_message(msg)) is not None]


# ── 主流程（RolloutRunner stdin/stdout 接口）─────────────────────────────

def _configure_logging(log_file: str | None) -> None:
    """
    在 main() 里调用，根据 --log-file 参数决定是否把 logger 输出同时写到文件。
    保留默认的 stderr handler（通过模块顶部的 basicConfig 已经装好），额外追加
    一个 FileHandler。这样 run_rollout.py 串行跑时可以 tail -f 看实时进度。

    参考 Deep_Research/agent_runner.py:259-272 的同类实现。
    """
    if not log_file:
        return
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(file_handler)
    root.setLevel(logging.INFO)


def main():
    # argparse 解析 CLI 参数（run_rollout.py 可通过 cmd 追加 --log-file）
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write INFO logs to file (in addition to stderr)",
    )
    args, _ = parser.parse_known_args()
    _configure_logging(args.log_file)

    payload = json.loads(sys.stdin.read())

    data_dir = payload.get("data_dir", "")

    # 用 data_dir 增强 question（与 thinkdepthai 一致：将数据位置追加到问题中）
    question = payload.get("question", "")
    if data_dir:
        question = (
            f"{question}\n\n## Data Location\n\n"
            f"The telemetry data for this incident is located at: `{data_dir}`\n"
            f"Start by calling `list_tables_in_directory(directory=\"{data_dir}\")` "
            f"to discover available parquet files."
        )

    # 构建 config（对应 AIRA 的 config["configurable"]）
    config = {
        "configurable": {
            "question": question,                       # augmented question（增强后）
            "user_prompt": payload["user_prompt"],
            "system_prompt": payload["system_prompt"],
            "data_dir": data_dir,
            "compress_system_prompt": payload["compress_system_prompt"],
            "compress_user_prompt": payload["compress_user_prompt"],
            "number_of_queries": 1,   # 生成 1 个综合调查查询
            "num_reflections": 2,     # 2 轮串行反思（对齐 AIRA 默认 reflection_count=2）
        }
    }

    agent = build_agent()

    # 初始状态（中间状态 causal_graph 替代了原 AIRA 的 running_summary text）
    initial_state = {
        "queries": [],
        "data_research_results": [],
        "schema_messages": [],
        "causal_graph": {"nodes": [], "edges": [], "root_causes": []},
        "accumulated_findings": [],
        "final_report": "",
        "all_tool_messages": [],
    }

    # 运行 AIRA 多阶段流水线
    result_state = agent.invoke(input=initial_state, config=config)

    # 输出结果
    output = strip_markdown_json(result_state.get("final_report", ""))
    all_tool_msgs = result_state.get("all_tool_messages", [])
    trajectory = convert_trajectory(all_tool_msgs)

    # Usage 采集：优先用 UsageTracker（monkey-patch OpenAI/Anthropic SDK），
    # 非 Claude 模型走 ChatOpenAI SDK，install_openai_hooks 可拦截
    usage = _tracker.get_usage()

    result = {"output": output, "trajectory": trajectory, "usage": usage}
    # 单行输出，runner._parse_last_json 从末行解析
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
