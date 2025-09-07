教程指南
=====================

🛠️ 安装指南
------------

在开始使用 NaturalCC 之前，请确保系统满足以下要求：

- GCC/G++ 版本不低于 5.0  
- NVIDIA GPU、NCCL 和 CUDA Toolkit（用于训练新模型，推荐）  
- NVIDIA Apex 库（可选，加速训练）  

环境配置步骤：

1. **可选：创建 Conda 环境**  
   建议用户使用 Conda 创建独立环境并激活，以便管理依赖。

2. **从源码构建 NaturalCC**  
   
   - 克隆项目仓库并进入目录。  
   - 安装项目依赖。  
   - 进入 `src` 目录，进行可编辑模式安装。  

3. **安装额外依赖**  
   
   - 安装音频相关依赖（libsndfile）。  
   - 安装最新版本的 HuggingFace Transformers 与 Accelerate 库。  

4. **HuggingFace Token（部分模型必需）**  
   对于如 StarCoder 这类模型，需要使用 HuggingFace 账号登录以获取访问权限。  

🚀 快速上手
-----------

**示例 1：代码生成**

1. 下载模型检查点（以 Codellama-7B 为例）。  
2. 准备测试数据集，格式为 JSON 文件，每个样本包含输入字段。  
3. 初始化任务并指定模型与设备（如 GPU）。  
4. 加载下载的模型权重。  
5. 加载数据集文件。  
6. 运行模型，得到结果并输出到指定路径。  

.. mermaid::

   graph TD

      A[下载模型检查点] --> B[准备测试数据集]
      B --> C[初始化任务并指定设备]
      C --> D[加载模型权重]
      D --> E[加载数据集文件]
      E --> F[运行模型并输出结果]

**示例 2：代码摘要（Code Summarization）**

1. 下载并处理数据集（例如 Python-Wan 数据集），包括下载、清洗、属性转换等步骤。  
2. 进行摘要任务相关的预处理，将代码与注释转化为合适格式。  
3. **注册自定义模型**：  
   
   - 将新模型加入 ``ncc/models`` 与 ``ncc/modules``。  
   - 如果训练策略较复杂，可在 ``ncc/criterions`` 与 ``ncc/trainers`` 中扩展。  
   - 别忘了在相应的 ``__init__.py`` 中注册模块。  
4. **训练与推理**：  
   
   - 根据任务选择模型与配置文件进行训练。  
   - 使用训练好的模型进行推理。  

📚 数据集支持
--------------

NaturalCC 提供了多样化的数据集支持，涵盖代码分析与处理的不同方面。包括但不限于：

- **Python（Wan 等人）**  
- **CodeSearchNet（Husain 等人）**  
- **CodeXGlue（Feng 等人）**  
- **Py150** 
- **OpenCL（Grewe 等人）**  
- **Java（Hu 等人）**  
- **Stack Overflow**  
- **DeepCS（Gu 等人）**  
- **AVATAR（Ahmad 等人）**  
- **StackOverflow（Iyer 等人）**  

更多任务与数据集的详细教程，请参考各模块自带的 README 文档。
