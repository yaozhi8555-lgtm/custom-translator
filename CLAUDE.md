# 项目说明：custom-translator

## 项目背景
高度定制化的翻译软件。不追求"一次生成可用软件"，而是通过一步步构建学习 Prompt Engineering、RAG、Tool Use、MCP、Agent 编排，最终具备为企业构建 Agent 系统的能力。

## 关键设计决策
- **API 调用**：走 OpenRouter（兼容 OpenAI 接口格式），非直连 Anthropic API。原因：统一计费入口、价格更划算。
- **依赖包**：`openai` + `python-dotenv`（不是 `anthropic` 包）
- **Provider 抽象层**：`src/translator/providers/` 下定义统一接口，当前只实现 OpenRouter，未来可扩展其他 provider
- **架构**：暂不做前后端分离，阶段0-5 全部 CLI，阶段6+ 视 MCP/Agent 需要再演进
- **成本控制**：每次调用记录 input/output tokens，缓存机制为非全量缓存（按标准化后的句子 + 模型/prompt版本作为 key）

## 当前阶段
**阶段1：核心直译功能**
详细步骤见 [`docs/phase1-steps.md`](docs/phase1-steps.md)

## 完整路线图
详见 `docs/翻译软件开发计划.md`（阶段0-8 全量设计文档）

## 工作原则
- 不要一次生成完整可用代码，需要逐步讲解，确保用户理解每一步在做什么
- 异常处理等到真实报错出现后再针对性补充，不预先假设所有情况
