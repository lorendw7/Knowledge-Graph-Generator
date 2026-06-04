# 知识图谱生成器（Knowledge Graph Generator）

使用大语言模型（GPT-4o）从一段自然语言文本中抽取实体与关系，构建知识图谱，并通过 [pyvis](https://pyvis.readthedocs.io/) 渲染为可交互的 HTML 网页。

本项目基于 LangChain 的 `LLMGraphTransformer`，演示了三种约束逐步加强的抽取方式：

1. **完全无约束** —— 节点类型与关系名都由 LLM 自由生成。
2. **节点类型白名单** —— 限定可出现的节点类型。
3. **节点 + 关系白名单** —— 进一步固定关系 schema，得到结构化最严格的图谱。

示例文本为英文维基百科中 **九州大学（Kyushu University）** 的条目。

## 环境要求

- Python 3.10 及以上
- 一个可访问 `gpt-4o` 的 OpenAI API Key

## 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/lorendw7/Knowledge-Graph-Generator.git
cd Knowledge-Graph-Generator

# 2. （可选）创建虚拟环境
python -m venv .venv
.venv\Scripts\activate     # Windows PowerShell
# source .venv/bin/activate  # macOS / Linux

# 3. 安装依赖
pip install -r requirements.txt
```

在项目根目录创建 `.env` 文件，写入：

```env
OPENAI_API_KEY=sk-...
```

> `.env` 已被 `.gitignore` 忽略，**请勿提交密钥到仓库**。

## 运行方式

本项目有两种使用方式：**Notebook**（逐步探索）或 **Streamlit 网页应用**（点点点的图形界面）。

### 方式 A —— Notebook

在 Jupyter 或 VS Code 中打开 `knowledge_graph.ipynb`，点击"全部运行"。Notebook 会依次：

1. 从 `.env` 读取 API Key；
2. 把示例文本送入 GPT-4o 进行实体/关系抽取；
3. 打印 `Node` / `Relationship` 列表；
4. 生成 `knowledge_graph.html` 并在浏览器中自动打开。

如需替换为自己的文本，修改 **"Extract graph data"** 下方代码格的 `text = """ ... """` 即可。

### 方式 B —— Streamlit 网页应用

运行 `app.py` 中定义的交互式网页应用：

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。在界面里可以：粘贴任意文本、选择模型、（可选）限制节点类型、点击**生成知识图谱**，即可查看交互式图谱以及原始的节点/关系数据——全程无需改代码。应用会从 `.env` 读取 `OPENAI_API_KEY`，也可以直接在侧边栏粘贴 Key。

> 需要 `streamlit >= 1.50`。若启动时遇到 `ImportError`，请参考 `requirements.txt` 中关于 `starlette` / `fastapi` 版本配合的说明。

## 工作原理

| 步骤 | 对应 Cell | 说明 |
|---|---|---|
| 加载密钥 | `dotenv` cell | 从 `.env` 中读取 `OPENAI_API_KEY` |
| 构建抽取管道 | `ChatOpenAI` + `LLMGraphTransformer` | 把 GPT-4o 接入 LangChain 的图抽取器 |
| 抽取 | `aconvert_to_graph_documents(documents)` | LLM 返回结构化的 `(节点, 关系)` 三元组 |
| 可视化 | `visualize_graph(...)` | 使用 pyvis 渲染交互式 HTML，采用 forceAtlas2 力导向布局 |

`visualize_graph` 函数会先过滤掉两端节点不在节点列表中的"悬空边"，再按节点 `type` 分组上色，方便区分实体类别。

## Schema 约束

约束 schema 可以让图谱更聚焦、可入库使用：

```python
allowed_nodes = ["Person", "Organization", "Location", "Award", "ResearchField"]
allowed_relationships = [
    ("Organization", "LOCATED_IN", "Location"),
    ("Organization", "MERGED_WITH", "Organization"),
    ("Person",       "ALUMNUS_OF", "Organization"),
    # ...
]
```

将以上参数传入 `LLMGraphTransformer(allowed_nodes=..., allowed_relationships=...)` 即可。

## 输出文件

`knowledge_graph.html` —— 一个自包含的交互式图谱网页，可拖拽节点、悬停查看类型。该文件在每次运行时会被覆盖，已加入 `.gitignore`。

## 许可证

MIT（如有 `LICENSE` 文件，以其为准）。

---

English version: see [README.md](./README.md).
