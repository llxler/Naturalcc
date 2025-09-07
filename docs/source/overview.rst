Naturalcc 是什么
==================

简介
----

Naturalcc 是一个面向代码理解与代码智能任务的序列建模工具包（toolkit），用于训练与评估大规模代码模型、做特征提取、benchmark 等。  
它旨在通过先进的机器学习技术弥合编程语言与自然语言之间的差距，可用于代码生成、代码补全、代码摘要、代码检索、代码克隆检测、类型推断等多种软件工程任务。

核心亮点（概览）
----------------

- 模块化、可扩展：基于 registry 机制，方便扩展任务与模型。
- 支持多种数据集与预处理脚本（如 CodeSearchNet、Py150、Python-Wan 等）。
- 支持大规模代码模型（例如 CodeLlama、CodeT5、CodeGen、StarCoder）。
- 支持分布式训练（torch.distributed / NCCL）与混合精度训练（FP16）。
- 丰富的评测与 benchmark 工具（pass@k、BLEU、ROUGE 等适配自 code 任务）。

目录结构建议
------------

项目文档结构（对应本 Sphinx 文档）：

- 总览
- 安装与快速上手
- 子功能模块文档：
  
  - 仓库级上下文提取与补全
  - C 语言图提取与提示词生成
  - Instruction Fine-tuning
  - Tokenizer 自定义与评估工具箱

📖 愿景
-------

**NaturalCC** 是一个序列建模工具包，旨在通过先进的机器学习技术弥合编程语言与自然语言之间的差距。  
它帮助研究人员与开发者训练自定义模型，支持多种软件工程任务，包括：代码生成、代码补全、代码摘要、代码检索、代码克隆检测与类型推断。

🌟 核心特性
-----------

- **模块化与可扩展框架**：基于 Fairseq 的 registry 机制，便于适配和扩展到多种代码任务。  
- **数据集与预处理工具**：内置多个经过清理与预处理的 benchmark（如 Human-Eval、CodeSearchNet、Python-Doc、Py150），并提供基于编译器（如 LLVM）的特征提取脚本。  
- **支持大规模代码模型**：内置 Code Llama、CodeT5、CodeGen、StarCoder 等主流大模型。  
- **评测与 Benchmark**：覆盖多种下游任务（代码生成、补全等），支持主流评测指标（如 pass@k）。  
- **高效训练优化**：依赖 NCCL 库与 torch.distributed，实现多 GPU 高效分布式训练，支持 FP32 与 FP16 混合精度加速训练与推理。  
- **增强的日志记录与调试支持**：提供详细的日志与反馈信息，帮助模型训练与调试过程中的性能优化。  

✨ 最新进展
-----------

- **2023-11-25**: NaturalCC 2.0 发布！兼容 Hugging Face Transformers，并支持 Code Llama、CodeT5、CodeGen、StarCoder 等模型。旧版本请参考 *ncc1* 分支。  
- **2023-04-19**: 集成论文 *"You See What I Want You to See: Poisoning Vulnerabilities in Neural Code Search"* 源码。  
- **2022-05-10**: 合并论文 *"What Do They Capture? - A Structural Analysis of Pre-Trained Language Models for Source Code"* 的源码。  
- **2022-01-25**: NaturalCC 工具包论文被 ICSE 2022 Demo Track 接收。  
