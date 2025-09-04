Tokenizer 自定义与评估工具箱
====================================

简介
----

Tokenizer 工具箱提供自定义分词器（Tokenizer）训练、词表管理、以及评估套件，用于验证不同 Tokenizer 在代码任务上的表现（如对代码片段的分割、subtoken 频率、OOV 率等）。

功能展示
--------

- 基于 BPE / SentencePiece / Unigram 的 Tokenizer 训练。
- 支持合并自定义 token（例如常见 API 名称、宏名）。
- 提供评估指标：token 长度分布、平均序列长度、OOV 比率、下游任务影响评估（可选）。

使用教程
--------

1. 训练 Tokenizer（示例，SentencePiece）：
   .. code-block:: shell

      python -m ncc.tools.tokenizer.train --input data/code_corpus.txt --model_prefix my_tokenizer --vocab_size 50000

2. 应用 Tokenizer：
   .. code-block:: python

      from ncc.tools.tokenizer import load_tokenizer
      tok = load_tokenizer("my_tokenizer.model")
      tokens = tok.encode("int foo(int a) { return a+1; }")

3. 评估：
   .. code-block:: shell

      python -m ncc.tools.tokenizer.evaluate --model my_tokenizer.model --dataset data/val_code.txt --out report.json

附加说明
--------

- 推荐在训练 Tokenizer 前先清洗代码语料（去掉过长注释、二进制文件等）。
- 对于跨语言场景，建议训练联合词表或使用 language-id 前缀。
