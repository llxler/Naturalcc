C 语言图提取并生成模型提示词
================================

简介
----

该模块专注于 C 语言的静态/半静态分析，能够从 C 源码中提取抽象语法树（AST）、控制流图（CFG）、数据流依赖图（DFG）等结构化信息，并将其转换成适用于语言模型的提示词（prompt）或结构化特征（structured features）。

功能展示
--------

- 支持基于 clang/LLVM 的 AST 提取。
- 生成 CFG/DFG 并提取关键路径或影响集合。
- 将图结构序列化为模型友好的格式（例如结构化概率矩阵、edge-list、flat prompt + metadata）。

展示示例（伪代码）
----------------------

.. code-block:: python

   from ncc.tools.c_graph import CGraphExtractor

   g = CGraphExtractor.parse_file("example.c")
   summary = CGraphExtractor.graph_to_prompt(g, focus_function="foo")
   # 将 summary 作为 prompt 输入到模型

使用教程
--------

1. 依赖：需要安装 clang / libclang（并配置 LD_LIBRARY_PATH 或类似变量）。
2. 运行提取：
   .. code-block:: shell

      python -m ncc.tools.c_graph --file example.c --out example_graph.json

3. 将 graph JSON 转换为 prompt：
   .. code-block:: python

      from ncc.tools.c_graph import graph_to_prompt
      prompt = graph_to_prompt("example_graph.json", focus="foo")
