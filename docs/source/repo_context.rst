仓库级别代码上下文提取与补全
================================

简介
----

本模块用于从代码仓库级别提取上下文信息（例如文件间调用关系、跨文件符号引用、函数/类上下文）并基于提取结果做补全或生成提示（prompts）。该功能适用于需要跨文件上下文的模型推理或训练场景。

该项目通过提取代码仓库的上下文信息，结合语义分析和相似度计算，生成高质量的代码补全建议，帮助开发者更高效地理解和编写代码。

主要功能
^^^^^^^^^^^^^

1. **仓库上下文提取**：
   
   - 通过解析代码仓库中的文件，构建 **控制依赖图（CCG）和语义图数据库**，提取代码片段之间的依赖关系和上下文信息。
   - 使用工具如Slicing和GraphDatabaseBuilder，对代码进行切片，生成前向依赖上下文、子图和编码信息。
   - 提供了跨文件的定义和引用上下文提取功能（如 ``construct_cross_file_definition_context``, ``construct_cross_file_reference_context``），帮助理解代码片段之间的跨文件依赖关系。
2. **语义相似度计算与上下文推荐**：
   
   - 实现了多种相似度计算方法（如编辑距离、Jaccard相似度、BM25、TF-IDF等），用于比较代码片段之间的语义相似性。
   - 通过SemanticReranking类，结合预训练模型（如CodeBERT、UniXcoder），对代码片段进行语义重排序，提升推荐结果的准确性。
3. **仓库级别代码补全**：
   
   - 通过构建查询子图(``build_query_subgraph``)和搜索上下文(``CodeSearchWorker``)，实现了基于上下文的代码补全。
   - 支持单阶段和双阶段的上下文查找策略，结合语义图和文本相似度，推荐最相关的代码片段。
4. **多语言支持与高效处理**：
   
   - 支持多种编程语言（如Python、Java、TypeScript等），并通过多进程和多线程优化了大规模代码仓库的处理效率。


功能展示
--------

- 从 Git 仓库解析文件依赖关系、符号引用。
- 为目标函数/片段构建上下文窗口（包含调用链、依赖文件片段）。
- 支持把上下文编码为结构化 prompt，直接供 LLM 进行补全/生成。

展示示例
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ncc.tools.repo_context import RepoContextExtractor

   extractor = RepoContextExtractor(repo_path="/path/to/repo")
   ctx = extractor.extract_context(file_path="src/foo.c", line_no=123, window=500)
   prompt = extractor.build_prompt(ctx, include_callers=True)
   # 将 prompt 传给模型进行补全

使用教程
--------

1. 准备：克隆目标仓库并确保依赖可解析（例如编译配置、include 路径等）。
2. 运行提取器：
   
   .. code-block:: shell

      python -m ncc.tools.repo_context --repo /path/to/repo --target src/foo.c:123 --out out.json

3. 输出解析后包含：调用链、跨文件引用、常量/宏定义上下文，格式为 JSON（示例见工具 README）。
4. 将输出转换为模型输入 prompt（工具提供 helper 函数）。

参数说明（常见）
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``--repo``: 仓库路径
- ``--target``: 目标位置，格式 ``file:lineno``
- ``--window``: 上下文窗口大小（字符或行）

仓库上下文提取测试
-------------------

本目录包含用于测试不同类型仓库上下文的代码。

目录结构
^^^^^^^^^

.. code-block::

   test_repo_context/
   ├── line_completion_dependency_graph.jsonl    # 依赖图生成的测试数据
   ├── line_completion_rg1_tfidf.jsonl           # TF-IDF相似性测试数据
   ├── test_common_similar_context.py            # 通用相似性上下文检索测试（支持
   ├── test_graph_semantic_similar_context.py    # 图语义结构相似性上下文检索测试
   └── test_structural_related_context.py        # 结构相关上下文检索测试

测试类型
^^^^^^^^^

1. **通用相似性测试** (``test_common_similar_context.py``)
   
   - 基于文本相似度的上下文匹配
   - 支持UnixCoder\CodeBert\BM25\TF-IDF\编辑相似度等多种相似度检索方法
   - 示例测试命令：基于TF-IDF的上下文检索

   .. code-block:: shell

      python augment_with_cfc.py \
      --language python \
      --rerank True \
      --ranker sparse \
      --ranking_fn tfidf \
      --crossfile_distance 100 \
      --maximum_chunk_to_rerank 1000 \
      --maximum_cross_files 1000 \
      --maximum_cross_file_chunk 10 \
      --use_next_chunk_as_cfc True \
      --skip_if_no_cfc False \
      --output_file_suffix rg1

2. **图语义相似性测试** (``test_graph_semantic_similar_context.py``)
   
   - 基于代码语义图的相似性分析
   - 考虑变量、函数调用等语义关系
   - 测试命令：``python test_graph_semantic_similar_context.py``

3. **结构相关性测试** (``test_structural_related_context.py``)
   
   - 分析代码结构相关性
   - 考虑类继承、接口实现等结构关系
   - 测试命令：``python test_structural_related_context.py``


仓库级别代码补全测试
--------------------

目录结构
^^^^^^^^^

.. code-block::

   repo-level_code_completion/
   ├── config.yaml                # accelerate配置文件
   ├── custom_generate.py         # 自定义生成逻辑
   ├── data_merge.py             # 数据合并工具
   ├── eval.py                   # 评估主程序
   ├── eval_metric.py            # 评估指标计算
   ├── eval_utils.py             # 评估工具函数
   ├── merged_line_completion.jsonl # 合并后的测试数据
   ├── repofuse-main/            # 主实验结果的输出目录
   │   └── Qwen2.5-Coder-7B-Instruct/ # 特定模型的评估结果
   ├── run_batch_experiments.py  # 实验运行脚本
   └── test.md                   # 测试文档

运行评估
^^^^^^^^^

.. code-block:: shell

   python run_batch_experiments.py --task repofuse-main

评估指标
^^^^^^^^^

- Exact Match (EM)
- Edit Similarity (ES)
- Identifier Match (IM)