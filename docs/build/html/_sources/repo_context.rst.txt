仓库级别代码上下文提取与补全
================================

简介
----

本模块用于从代码仓库级别提取上下文信息（例如文件间调用关系、跨文件符号引用、函数/类上下文）并基于提取结果做补全或生成提示（prompts）。该功能适用于需要跨文件上下文的模型推理或训练场景。

功能展示
--------

- 从 Git 仓库解析文件依赖关系、符号引用。
- 为目标函数/片段构建上下文窗口（包含调用链、依赖文件片段）。
- 支持把上下文编码为结构化 prompt，直接供 LLM 进行补全/生成。

展示示例
------------------

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
- ``--repo``: 仓库路径
- ``--target``: 目标位置，格式 ``file:lineno``
- ``--window``: 上下文窗口大小（字符或行）
