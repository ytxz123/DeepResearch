# Deep Research Agent

> 输入一个研究问题,自动产出一份带引用、可审计的深度调研报告。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.13-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-1C3C3C.svg)](https://www.langchain.com/langgraph)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9.svg)](https://docs.astral.sh/uv/)

基于 **LangGraph + LangChain** 的多智能体深度调研系统。给出一个开放式问题,系统会自动拆解子主题、并行联网查证、迭代精修草稿,最终产出带引用编号与参考文献的 Markdown 报告。

---

## 快速开始

**环境要求**:Python ≥ 3.13、[uv](https://docs.astral.sh/uv/)、一个 [DeepSeek](https://platform.deepseek.com/) Key、一个 [Tavily](https://tavily.com/) 搜索 Key。

```bash
git clone https://github.com/ytxz123/DeepResearch.git
cd DeepResearch
uv sync

cp .env.example .env       # 填入 API Key，无需改任何 YAML
uv run python run.py "帮我写一份关于英伟达最新 GPU 的调研报告"
```

调研耗时随深度档位变化较大（`quick` 约 5–10 分钟，`standard` 约 30–60 分钟），完成后报告保存到 `results/output_report_YYYY-MM-DD.md`,并在终端打印开头摘要。

常用选项:

```bash
uv run python run.py                                   # 交互式输入问题
uv run python run.py "问题" -o report.md                # 指定输出文件
uv run python run.py "问题" --depth quick                # 快速档（默认 standard）
uv run python run.py "问题" --no-clarify                 # 关闭开跑前的追问
uv run python run.py "问题" --print-report               # 终端打印报告全文
uv run python run.py "问题" --log-level DEBUG            # 查看详细日志
```

**调研深度**：`--depth` 一条指令同时控制主管决策轮数与并行子代理数。

| 档位 | 主管轮数 | 并行子代理 | 适用场景 |
|---|---|---|---|
| `quick` | 3 | 2 | 先摸清方向，几分钟出结果 |
| `standard`（默认） | 6 | 3 | 常规调研 |
| `deep` | 10 | 3 | 尽可能穷尽，耗时与检索额度消耗最高 |

> 轮数不宜设高：报告质量通常在前几轮就收敛，之后的迭代多为重复检索。每一轮由主管自行决定派发检索还是精修草稿，一轮里可以并行派发多个子代理，因此轮数不等于检索次数。

**开跑前追问**：需求含糊时（例如只给一句「帮我调研一下 AI」），系统会先反问你一句再开始，最多追问 2 轮；信息已经足够时不会打断。追问依赖检查点，通过 `--no-clarify` 可关闭；以编程方式调用图时默认不启用。

---

## 工作流程

```
用户提问
   │
   ▼
① clarify                需求含糊时先反问（可用 --no-clarify 关闭）
② write_research_brief   问题 → 可执行的调研提纲
③ write_draft_report     提纲 → 第一版草稿
④ Supervisor 研究循环（核心，嵌套子图）
     Supervisor 决策 ──┬─ think_tool           反思进展与信息缺口
                       ├─ ConductResearch ×N   并行派发子研究（最多 3 个）
                       ├─ refine_draft_report  用新发现精修草稿
                       │    └─ Evaluator 三维打分（全面性/准确性/一致性）
                       └─ Red Team 对抗审查 ── 缺陷回注下一轮研究
                          ↓ ResearchComplete / 达到迭代上限
⑤ final_report_generation   综合全部发现，带引用成稿
   │
   ▼
Markdown 调研报告（章节结构 + 引用编号 + 参考文献）
```

> **一轮只精修一次**：主管一轮里可能发出多个 `refine_draft_report` 调用，但该工具的参数由框架注入、调用之间没有差异，系统只执行一次并复用结果。

| 角色 | 职责 |
|---|---|
| **Supervisor** | 拆解问题、并行委派子研究、判断何时收尾 |
| **Researcher** | 负责单个子主题,反复「搜索 → 反思」直至信息饱和 |
| **Evaluator** | 从全面性 / 准确性 / 一致性三维打分(0–10),低分触发修复 |
| **Red Team** | 对抗性审查草稿,寻找逻辑漏洞与盲区 |
| **Writer** | 负责简报、草稿、精修与最终报告的成文 |

---

## 配置

密钥可以写在 `.env`(推荐)或直接填进 `config/*.yml`。优先级:**真实环境变量 > `.env` > YAML 默认值**。

| 环境变量 | 作用 | 默认值 |
|---|---|---|
| `DEEPSEEK_API_KEY` | LLM 密钥 | — |
| `TAVILY_API_KEY` | 搜索密钥 | — |
| `CONFIG_PATH` | 配置文件路径 | `config/deepseek.yml` |
| `STAGE` | 使用的 stage | `prod` |
| `RESEARCH_DEPTH` | 调研深度档位（等价 `--depth`） | `standard` |
| `DEEP_RESEARCH_LOG_LEVEL` | 日志等级 | `INFO` |

> ⚠️ `.env` 与 `config/*.yml` 中请勿提交真实密钥。

**接入其他厂商**:改 `config/deepseek.yml` 里的 `base_url` 与各角色的 `handle` 即可指向任意 OpenAI 兼容接口 —— `llm.py` 已固定 `model_provider="openai"`,换厂商无需改代码。若目标厂商支持 `json_schema`,删掉 `structured_output_method: json_mode` 那一行。

**角色级模型分配**:`stages.prod.roles` 下每个角色可单独指定 `handle` / `timeout_seconds`,便于在质量与成本间取平衡(如主管用强模型、摘要用轻量模型)。

> 关于 `max_tokens`:DeepSeek 模型是思考模型，思维链同样计入输出上限，撞限时返回的是空正文而非截断正文；且 langchain 发出的是 DeepSeek 不认的 `max_completion_tokens`。因此配置里刻意不设该值，交由服务端上限兜底。

**研究循环规模**由深度档位决定，档位表见上文。另有 `min_need_repair_score`(默认 6.0)：草稿三维均分低于该值时，Evaluator 会提醒主管在下一轮修复。

**接入自定义搜索后端**:实现 `deep_research.tools.search_factory.SearchProvider` 协议(`build_client` / `search` / `defaults`)并调用 `register_provider` 注册,再把配置里的 `search.backend` 指向它。`deep_research/providers/` 提供了模板。

---

## 项目结构

```
Deep_Research/
├── config/                  # deepseek.yml
├── deep_research/
│   ├── agent_builder.py     # 主工作流（追问 → 简报 → 草稿 → 研究循环 → 成稿）
│   ├── llm.py               # LLM 客户端工厂（按角色解析配置）
│   ├── utils.py             # 配置加载 / 环境变量展开
│   ├── agents/              # supervisor / research / evaluator / red_team / draft
│   ├── states/              # LangGraph State 与结构化输出 Schema
│   ├── prompts/             # 各角色提示词
│   ├── tools/               # 搜索 / 反思 / 精修工具 + 可插拔后端
│   └── providers/           # 自定义搜索后端模板
├── run.py                   # CLI 入口
├── .env.example             # 密钥模板
└── pyproject.toml
```

---

## License

[MIT](LICENSE)
