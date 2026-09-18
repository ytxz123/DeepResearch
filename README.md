# Deep Research Agent

> 输入一个研究问题,自动产出一份带引用、可审计的深度调研报告。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.13-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-1C3C3C.svg)](https://www.langchain.com/langgraph)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9.svg)](https://docs.astral.sh/uv/)

基于 **LangGraph + LangChain** 的多智能体深度调研系统。给出一个开放式问题,系统会自动拆解子主题、并行联网查证、迭代精修草稿,最终产出带引用编号与参考文献的 Markdown 报告。

---

## 快速开始

**环境要求**:Python ≥ 3.13、[uv](https://docs.astral.sh/uv/)、一个 OpenAI 兼容的 LLM Key([Qwen](https://bailian.console.aliyun.com/) 或 [DeepSeek](https://platform.deepseek.com/))、一个 [Tavily](https://tavily.com/) 搜索 Key。

```bash
git clone https://github.com/ytxz123/DeepResearch.git
cd DeepResearch
uv sync

cp .env.example .env       # 填入 API Key，无需改任何 YAML
uv run python run.py "帮我写一份关于英伟达最新 GPU 的调研报告"
```

调研约需 **10–20 分钟**,完成后报告保存到 `results/output_report_YYYY-MM-DD.md`,并在终端打印开头摘要。

常用选项:

```bash
uv run python run.py                                   # 交互式输入问题
uv run python run.py "问题" -o report.md                # 指定输出文件
uv run python run.py "问题" --depth quick                # 快速档（默认 standard）
uv run python run.py "问题" --config config/qwen.yml     # 换用 Qwen（默认 DeepSeek）
uv run python run.py "问题" --no-clarify                 # 关闭开跑前的追问
uv run python run.py "问题" --print-report               # 终端打印报告全文
uv run python run.py "问题" --log-level DEBUG            # 查看详细日志
```

**调研深度**：`--depth` 一条指令同时控制研究轮数与并行子代理数。

| 档位 | 研究轮数 | 并行子代理 | 适用场景 |
|---|---|---|---|
| `quick` | 3 | 2 | 先摸清方向，几分钟出结果 |
| `standard`（默认） | 6 | 3 | 常规调研 |
| `deep` | 10 | 3 | 尽可能穷尽，耗时与检索额度消耗最高 |

> 轮数不宜设高：报告质量通常在前几轮就收敛，之后的迭代多为重复检索。

**开跑前追问**：需求含糊时（例如只给一句「帮我调研一下 AI」），系统会先反问你一句再开始，最多追问 3 轮；信息已经足够时不会打断。追问依赖检查点，通过 `--no-clarify` 可关闭；以编程方式调用图时默认不启用。

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
| `QWEN_API_KEY` / `DEEPSEEK_API_KEY` | LLM 密钥 | — |
| `TAVILY_API_KEY` | 搜索密钥 | — |
| `CONFIG_PATH` | 配置文件路径 | `config/deepseek.yml` |
| `STAGE` | 使用的 stage | `prod` |
| `RESEARCH_DEPTH` | 调研深度档位（等价 `--depth`） | `standard` |
| `DEEP_RESEARCH_LOG_LEVEL` | 日志等级 | `INFO` |

> ⚠️ `.env` 与 `config/*.yml` 中请勿提交真实密钥。

**切换厂商**:`config/qwen.yml` 与 `config/deepseek.yml` 内置两套配置,用 `--config` 切换。也可改 `base_url` 与各角色的 `handle` 接入任意 OpenAI 兼容接口 —— `llm.py` 已固定 `model_provider="openai"`,换厂商无需改代码。

**角色级模型分配**:`stages.prod.roles` 下每个角色可单独指定 `handle` / `max_tokens` / `timeout_seconds`,便于在质量与成本间取平衡(如主管用强模型、摘要用轻量模型)。

**研究循环规模**由深度档位决定，档位表见上文。另有 `min_need_repair_score`(默认 6.0)：草稿三维均分低于该值时，Evaluator 会提醒主管在下一轮修复。

**接入自定义搜索后端**:实现 `deep_research.tools.search_factory.SearchProvider` 协议(`build_client` / `search` / `defaults`)并调用 `register_provider` 注册,再把配置里的 `search.backend` 指向它。`deep_research/providers/` 提供了模板。

---

## 项目结构

```
Deep_Research/
├── config/                  # qwen.yml / deepseek.yml
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

## 常见问题

**Q: `No config found for stage 'prod'` / `Role 'xxx' not found`?**
A: 确认 `CONFIG_PATH` 指向的配置文件存在,且其中已配置该角色(需同时有 `backend` 与 `handle`)。

**Q: Qwen 报 401 / 403?**
A: 检查 Key 是否与 `base_url` 同地域(中国大陆 `dashscope.aliyuncs.com` ↔ 中国大陆 Key)。

**Q: DeepSeek 报 `This response_format type is unavailable now`?**
A: DeepSeek 不支持 `json_schema`,配置中需有 `structured_output_method: json_mode`(`config/deepseek.yml` 已内置)。

**Q: 报告写一半就断、或某一步返回空内容?**

A: DeepSeek 的模型是**思考模型**,先输出 `reasoning_content` 再输出正文,且思维链 token 同样计入 `max_tokens`。`max_tokens` 配得太小(例如 100)时预算会被思维链吃光,`content` 返回空串(`finish_reason=length`),上层只能拿到空消息。`config/deepseek.yml` 已为写长文的角色留出余量,自行调小时请留意。

**Q: 调研太慢?**
A: 用 `--depth quick` 跑快速档;或调低搜索的 `max_results`。

**Q: 报告语言?**
A: 自动跟随提问语言,用中文 / 英文提问即可。

---

## License

[MIT](LICENSE)
