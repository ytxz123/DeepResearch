# Deep Research Agent · 多智能体深度调研系统

> 输入一个研究问题,自动产出一份带引用、可审计、深度详实的调研报告。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.13-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-1C3C3C.svg)](https://www.langchain.com/langgraph)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9.svg)](https://docs.astral.sh/uv/)

`Deep Research Agent` 是一个基于 **LangGraph + LangChain** 的多智能体深度调研系统。它把「检索 → 并行分析 → 迭代精修 → 质量自检」组装成一套可编排、可迭代、可自我纠错的流水线:用户只需给出一个开放式研究问题,系统会自动拆解为多个子主题、并行联网查证、反复打磨草稿,最终生成结构清晰、附引用编号与参考文献的 Markdown 调研报告。

| 技术栈 | |
|---|---|
| 编排框架 | [LangGraph](https://www.langchain.com/langgraph) 1.x |
| LLM 生态 | [LangChain](https://www.langchain.com/) / LangChain-Core / langchain-openai |
| 默认 LLM | [Qwen](https://help.aliyun.com/zh/model-studio)(阿里云百炼 OpenAI 兼容接口)· 可一键切换 [DeepSeek](https://platform.deepseek.com/) |
| 联网检索 | [Tavily](https://tavily.com/)(可插拔搜索后端) |
| 数据校验 | Pydantic v2(结构化输出) |
| 运行环境 | Python ≥ 3.13 · 包管理 [uv](https://docs.astral.sh/uv/) |

---

## ✨ 核心特性

- **多智能体编排** —— Supervisor 统一调度,并行派发多个独立上下文的 Research Agent 分主题查证,各司其职、互不干扰。
- **「降噪式」迭代研究** —— 反复「提出新问题 → 检索增量 → 精修草稿」,每轮清除草稿中的不准确与不完整,逐步收敛到高质量报告。
- **自我进化质量闭环** —— Evaluator 从全面性 / 准确性 / 一致性三维打分,低于阈值自动触发修复;Red Team 对抗性审查寻找逻辑漏洞与盲区,缺陷自动注入下一轮研究。
- **上下文与成本可控** —— 网页长文自动摘要、结果 URL 去重、研究笔记压缩、独立上下文窗口,以及多层次的迭代 / 并发 / 搜索预算。
- **工程化可落地** —— 配置驱动(`config/` 下集中管理角色、模型、超时与搜索参数)、可插拔搜索后端、Pydantic 结构化输出、集中日志与失败兜底。

---

## 🏗️ 系统架构

### 总体流程

```
用户提问
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│ ① write_research_brief    ── 把问题转成可执行的调研提纲        │
│ ② write_draft_report      ── 基于简报产出第一版草稿            │
│ ③ Supervisor 研究循环(核心 · 嵌套子图)                        │
│      ┌─────────────────────────────────────────────┐          │
│      │  Supervisor 决策(LLM + 4 工具)               │          │
│      │      │ 委派研究 / 精修草稿                    │          │
│      │      ▼                                       │          │
│      │  supervisor_tools                            │          │
│      │   · think_tool          → 反思研究进展        │          │
│      │   · ConductResearch ×N  → 并行 Research Agent │          │
│      │   · refine_draft_report → Evaluator 三维打分  │          │
│      │      ▼                                       │          │
│      │  Red Team 对抗审查 ──(缺陷注入)──┐             │          │
│      │      └──────────────────────────┘             │          │
│      └──────────────────┬──────────────────────────┘          │
│                         │ ResearchComplete / 达到迭代上限      │
│ ④ final_report_generation ── 综合全部发现,带引用成稿           │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
调研报告(Markdown · 章节结构 + 引用编号 + 参考文献)
```

### 智能体角色

| 角色 | 模块 | 职责 |
|---|---|---|
| **Supervisor** 研究主管 | `agents/supervisor.py` | 拆解问题、决策下一步、委派并行任务、判断何时收尾;持有 `ConductResearch` / `ResearchComplete` / `think_tool` / `refine_draft_report` 四个工具 |
| **Researcher** 研究员 | `agents/research_agent.py` | 负责单个子主题,反复「搜索 → 反思」直至信息饱和,再压缩研究结果交回 Supervisor |
| **Evaluator** 质量评委 | `agents/evaluator_agent.py` | 从全面性 / 准确性 / 一致性三维打分(0–10),低分触发修复,并记录质量历史 |
| **Red Team** 红队 | `agents/red_team_agent.py` | 对抗性审查草稿,寻找逻辑缺陷、偏题与盲区;输出结构化批评,无问题则 PASS |
| **Writer** 写作 | `agents/draft_agent.py` | 负责研究简报、报告草稿、精修与最终报告的成文 |

---

## 📁 项目结构

```
Deep_Research/
├── config/                        # 配置目录
│   ├── qwen.yml                   # Qwen(阿里云百炼)配置
│   └── deepseek.yml               # DeepSeek 配置
├── deep_research/
│   ├── agent_builder.py           # 主工作流 StateGraph(4 阶段流水线)
│   ├── llm.py                     # LLM 客户端工厂(按角色从 config 解析)
│   ├── utils.py                   # 配置加载 / 日期等工具函数
│   ├── logging.py                 # 集中式日志配置
│   ├── agents/                    # 各智能体实现
│   ├── states/                    # LangGraph State 与结构化输出 Schema
│   ├── prompts/                   # 各角色的提示词库
│   ├── tools/
│   │   ├── tool.py                # 搜索 / 反思 / 精修工具实现
│   │   └── search_factory.py      # 可插拔搜索后端:Provider 协议 + 注册中心
│   └── providers/                 # 自定义搜索后端模板
├── run.py                         # CLI 启动入口
├── pyproject.toml                 # 项目依赖与元数据
├── uv.lock                        # 依赖锁定文件
└── LICENSE
```

---

## 🚀 快速开始

### 环境要求

| 依赖 | 要求 |
|---|---|
| Python | ≥ 3.13 |
| [uv](https://docs.astral.sh/uv/) | ≥ 0.5(依赖与虚拟环境管理) |
| LLM API | Qwen / DeepSeek 账号(OpenAI 兼容接口) |
| 搜索 API | [Tavily](https://tavily.com/) 账号 |
| LangSmith(可选) | 用于链路追踪与调试 |

### 1. 克隆并安装依赖

```bash
git clone https://github.com/ytxz123/DeepResearch.git
cd DeepResearch
uv sync
```

> `uv sync` 会依据 `pyproject.toml` 与 `uv.lock` 创建 `.venv` 并安装全部依赖。
> 若尚未安装 uv,可执行 `pip install uv` 或参考 uv 官方文档。

### 2. 配置密钥

编辑 `config/qwen.yml`(或 `config/deepseek.yml`),填入你的密钥:

```yaml
stages:
  prod:
    cognition:
      openai:
        base_url: https://dashscope.aliyuncs.com/compatible-mode/v1   # 百炼 OpenAI 兼容地址
        api_key: sk-xxx                        # ← 你的百炼 API Key(与地域绑定)
        default_model: qwen3.8-max
    search:
      backend: tavily
      tavily:
        api_key: tvly-xxx                      # ← 你的 Tavily API Key
```

> ⚠️ `config/*.yml` 已被 git 跟踪,**请勿把真实密钥提交到仓库**。若担心误提交,可用 `git update-index --skip-worktree config/qwen.yml` 让本地改动不被跟踪。

### 3. 运行一次深度调研

```bash
# 通义千问
uv run python run.py "帮我写一份关于英伟达最新 GPU 的调研报告" --config config/qwen.yml

# DeepSeek
uv run python run.py "2026 年多模态大模型的进展如何?" --config config/deepseek.yml
```

常用选项:

```bash
uv run python run.py                    # 不带参数进入交互式输入
uv run python run.py "问题" -o report.md # 指定输出文件
uv run python run.py "问题" --print-report  # 终端打印报告全文
uv run python run.py "问题" --log-level DEBUG
```

调研通常需要 **10–20 分钟**(取决于问题复杂度与迭代轮数)。完成后报告自动保存为 `results/output_report_YYYY-MM-DD.md`,并在终端打印开头摘要。

报告包含清晰的章节结构(`#` / `##` / `###`)、正文引用编号 `[1][2]…`、结尾参考文献列表,以及视问题类型而定的汇总对照表。

### 4.(可选)配置 LangSmith 追踪

在 [LangSmith](https://smith.langchain.com/) 中查看完整链路:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_API_KEY=lsv2_pt_xxx
export LANGSMITH_PROJECT=DeepResearch
```

---

## 🛠️ 配置详解

系统通过 `config/` 下的配置文件驱动,`stages` 下可定义多个环境(默认 `prod`)。配置文件通过 `--config` 参数或 `CONFIG_PATH` 环境变量选择,环境通过 `--stage` 或 `STAGE` 选择。

| 环境变量 | 作用 | 默认值 |
|---|---|---|
| `CONFIG_PATH` | 指定配置文件路径 | `config/deepseek.yml` |
| `STAGE` | 选择使用哪个 stage | `prod` |
| `DEEP_RESEARCH_LOG_LEVEL` | 日志等级 | `INFO` |

### 角色配置(`stages.<stage>.roles`)

每个角色独立配置模型,便于在质量与成本间取平衡(如主管用强模型、摘要用轻量模型):

```yaml
roles:
  supervisor:
    backend: openai          # 后端类型(当前支持 openai 兼容接口)
    handle: qwen3.8-max      # 模型名
    max_tokens: 131072       # 输出上限(可选)
    timeout_seconds: 600     # 请求超时
```

可用角色:`supervisor` / `researcher_main` / `researcher_summarizer` / `researcher_compressor` / `writer` / `draft` / `evaluator` / `red_team` / `context_pruner`(预留)。

### 搜索配置(`stages.<stage>.search`)

```yaml
search:
  backend: tavily             # 搜索后端
  tavily:
    api_key: tvly-xxx
    max_results: 3            # 单次搜索返回条数
    topic: general            # general / news / finance
    include_raw_content: true # 是否抓取原始网页内容(用于摘要)
    timeout_seconds: 300
```

### LLM 接入:Qwen / DeepSeek

通过 **OpenAI 兼容接口**接入任意 LLM;`llm.py` 已显式固定 `model_provider="openai"`,模型名无需特定前缀。

| 厂商 | base_url | 常用模型 |
|---|---|---|
| **Qwen**(阿里云百炼) | `https://dashscope.aliyuncs.com/compatible-mode/v1`(国际版 `dashscope-intl`) | `qwen3.8-max` / `qwen3.8-flash` |
| **DeepSeek** | `https://api.deepseek.com` | `deepseek-v4-pro` / `deepseek-v4-flash` |

`cognition.openai` 支持字段:

```yaml
cognition:
  openai:
    base_url: <兼容接口地址>        # 必填
    api_key: <你的 key>             # 必填
    default_model: <默认模型>       # 可被 roles.*.handle 覆盖
    temperature: 0                  # 可选,结构化任务建议 0
    timeout_seconds: 600            # 可选
    structured_output_method: json_mode  # 可选,不支持 json_schema 的厂商需设置
    extra_body:                     # 可选,厂商私有参数,原样透传
      enable_thinking: false        # Qwen 示例:关闭思考模式
```

> **思考类模型提示**:部分推理模型默认开启链式思考,可能影响工具调用 / JSON 输出稳定性,建议通过 `extra_body.enable_thinking: false` 关闭。

### 关键常量

| 常量 | 默认值 | 含义 | 位置 |
|---|---|---|---|
| `max_researcher_iterations` | 15 | 研究循环最大迭代轮数 | `agents/supervisor.py` |
| `max_concurrent_researchers` | 3 | 单轮最多并行子 Agent 数 | `agents/supervisor.py` |
| `min_need_repair_score` | 6.0 | 草稿均分低于该值触发修复 | `agents/supervisor.py` |
| `MAX_CRITIC` | 3 | 红队最大批评次数 | `agents/red_team_agent.py` |
| `MIN_DRAFT_LEN` | 50 | 草稿过短时跳过红队审查 | `agents/red_team_agent.py` |

---

## 🧠 工作原理

1. **研究简报** —— `write_research_brief` 把用户问题转译为具体、可执行且不臆测的调研提纲(结构化输出),并区分「研究范围」与「用户偏好」。
2. **报告草稿** —— `write_draft_report` 基于简报写出第一版结构化草稿,作为后续精修迭代的起点。
3. **研究循环** —— Supervisor 依据「降噪算法」提示词工作:
   - 通过 `think_tool` 反思研究进展与信息缺口;
   - 通过 `ConductResearch` 把每个子主题**并行委派**给独立的 Research Agent(最多 3 个并行,各自拥有独立上下文);
   - 子 Agent 反复「`tavily_search` 检索 → `think_tool` 反思」,信息饱和后压缩研究结果(去冗余、逐字保留事实与来源)交回;
   - 通过 `refine_draft_report` 用新发现精修草稿,随后 **Evaluator 三维打分**,低于阈值标记修复;
   - **Red Team** 独立审查草稿,将缺陷以结构化批评注入 Supervisor 的系统消息,驱动下一轮针对性补查 / 修正;
   - 当 `ResearchComplete` 被调用(或达到迭代上限)时结束,输出最终研究笔记。
4. **最终报告** —— `final_report_generation` 综合全部研究笔记与草稿,产出带引用编号、章节结构与参考文献的正式报告。

---

## ❓ 常见问题

**Q: 提示 `No config found for stage 'prod'`?**
A: 确认 `config/qwen.yml` 或 `config/deepseek.yml` 存在,并通过 `--config` 或 `CONFIG_PATH` 指向它。

**Q: 提示 `Role 'xxx' not found for stage`?**
A: 确认配置文件的 `roles` 中已配置该角色,且 `backend` 与 `handle` 字段均存在。

**Q: 如何切换 / 更换大模型?**
A: 修改 `cognition.openai.base_url` 与各角色的 `handle` 即可;`llm.py` 已固定 `model_provider="openai"`,任何 OpenAI 兼容接口都无需改动代码。

**Q: Qwen(百炼)调用报 401 / 403?**
A: 确认 `api_key` 正确,且与 `base_url` 同地域(中国大陆 `dashscope.aliyuncs.com` 对应中国大陆 Key;国际版 `dashscope-intl` 对应国际版 Key)。

**Q: DeepSeek 报 `This response_format type is unavailable now`?**
A: DeepSeek 不支持 `json_schema`,需在配置中设置 `structured_output_method: json_mode`(`config/deepseek.yml` 已内置)。

**Q: DeepSeek 工具调用异常?**
A: 若使用 `reasoner` 思考模型,其 `reasoning_content` 回传逻辑本工作流未处理,建议改用 `deepseek-v4-flash` / `deepseek-v4-pro`。

**Q: 如何接入其他搜索源(新闻 / 学术等)?**
A: 实现 `deep_research.tools.search_factory.SearchProvider` 协议(`build_client` / `search` / `defaults`),调用 `register_provider` 注册,并在配置中将 `search.backend` 指向该后端。`deep_research/providers/` 提供了模板。

**Q: 调研耗时太长?**
A: 调低 `max_researcher_iterations`、`max_concurrent_researchers` 或 `max_results`。

**Q: 报告语言如何控制?**
A: 系统会自动使用与提问相同的语言撰写报告,用中文 / 英文提问即可。

---

## 🤝 参与贡献

欢迎提交 Issue 与 Pull Request。

1. Fork 本仓库并创建分支:`git checkout -b feature/your-feature`
2. 提交改动:`git commit -m "feat: your feature"`
3. 推送并开启 Pull Request

---

## 📜 License

本项目采用 [MIT License](LICENSE) 开源。
