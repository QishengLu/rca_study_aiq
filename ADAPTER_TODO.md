# Adapter TODO — mirror ThinkDepthAI template

This framework was copied from `/home/nn/SOTA-agents/<orig-dir>/` as an
initial snapshot. To finalize WP3.x:

## Steps (run in this directory)

### 1. Copy canonical prompt (bit-identical to ThinkDepthAI)

```bash
mkdir -p src/<pkg>/prompts/agents/langgraph
cp /home/nn/sota-rca-study/sdk/sota_rca/prompts/rca.yaml \
   src/<pkg>/prompts/agents/langgraph/rca.yaml
```

### 2. Add PromptManager (copy from sdk)

```bash
mkdir -p src/<pkg>/prompts
cp /home/nn/sota-rca-study/sdk/sota_rca/prompts/manager.py \
   src/<pkg>/prompts/manager.py
```

### 3. Write `src/<pkg>/agents/eval_agent.py` (aegis v3 BaseAgent subclass)

Template:

```python
from datetime import datetime
from rcabench_platform.v3.sdk.llm_eval.agents.base_agent import BaseAgent, AgentResult
from rcabench_platform.v3.sdk.llm_eval.trajectory import Trajectory
from ..prompts.manager import PromptManager
from ..agent import <CoreAgent>          # this framework's existing core
from ..converters import build_trajectory # implement separately

class <PascalName>EvalAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "<framework-name>"

    async def run(self, incident: str, data_dir: str, **kwargs) -> AgentResult:
        prompts = PromptManager.get_prompts(__file__.replace(
            "agents/eval_agent.py", "prompts/agents/langgraph/rca.yaml"
        ))
        date = datetime.now().strftime("%Y-%m-%d")
        sp = prompts["RCA_ANALYSIS_SP"].format(date=date)
        up = prompts["RCA_ANALYSIS_UP"].format(incident_description=incident)
        # ... run framework with sp / up ...
        # collect framework messages -> v3 Trajectory
        traj = build_trajectory(framework_messages, agent_name=self.name(), system_prompt=sp)
        return AgentResult(response=final_json, trajectory=traj, metadata={...})
```

### 4. Write `src/<pkg>/converters.py` — framework-specific Trajectory builder

Goal: emit `Trajectory(agent_trajectories=[AgentTrajectory(turns=[Turn(messages=[Message(...)])])])`

For LangChain frameworks (thinkdepthai, aiq): reuse aegis v3's
`TrajectoryConverter.from_langchain_messages()`.

For TaskWeaver: walk `Post` objects → `Message(role, content, tool_calls=...)`.

For OpenRCA: parse Controller/Executor message pairs.

For ClaudeCode: parse stream-json events.

For mABC: emit nested SubAgentCall:
  - Main `ProcessScheduler` agent → `AgentTrajectory(trajectory_id="main")`
  - 6 experts → `AgentTrajectory(sub_agent_call_id="task_N")`

### 5. Register via setuptools entry-point

Edit `pyproject.toml`:
```toml
[project.entry-points."llm_eval.agents"]
<framework-name> = "<pkg>.agents.eval_agent:<PascalName>EvalAgent"
```

### 6. Install + smoke

```bash
cd frameworks/<framework-name>
uv pip install -e .
uv run rca llm-eval show-agents | grep <framework-name>   # should appear

# 1 case smoke (from repo root)
uv run python scripts/run_matrix.py configs/matrix/demo.yaml \
    --frameworks <framework-name> --models qwen36 --dry-run
```

### 7. Replace base_url / model hardcoding with UTU_LLM_* reads

Grep for hardcoded URLs / model names in `src/<pkg>/`:
```bash
grep -rn "shubiaobiao\|dashscope\|api\.openai\|claude-sonnet" src/
```
Replace each with reads from `UTU_LLM_BASE_URL` / `UTU_LLM_MODEL`.

### 8. Top of agent_runner / eval_agent: trigger UsageTracker

```python
from sota_rca.tracker import auto_install
auto_install()   # MUST be before any LLM SDK import
```

### 9. Cross-platform fixes

```bash
grep -rn "/home/nn" src/
# Replace each with Path(__file__).resolve().parent... or env var
```

### Verification

- [ ] `diff src/<pkg>/prompts/agents/langgraph/rca.yaml /home/nn/sota-rca-study/sdk/sota_rca/prompts/rca.yaml` produces NO diff
- [ ] `pip install -e .` succeeds
- [ ] `uv run rca llm-eval show-agents` lists this framework
- [ ] 1 case smoke produces `Trajectory.to_json()` writable to DB
- [ ] Dashboard renders the trajectory in CaseDetail
