import gradio as gr
from textwrap import dedent

APP_TITLE = "NaturalCC — Natural Code Comprehension"
APP_DESC = dedent(
    r'''
    # 📖 愿景
    **NaturalCC** 是一个序列建模工具包，旨在通过先进的机器学习技术弥合编程语言与自然语言之间的差距。

    它帮助研究人员和开发者训练自定义模型，用于多种软件工程任务：
    - 代码生成 / 自动补全 / 摘要生成
    - 代码检索与克隆检测
    - 类型推断

    ### 🌟 主要特性
    - **模块化与可扩展性**：基于 Fairseq 的注册机制，便于在不同的软件工程任务中扩展与适配。
    - **数据集与预处理工具+**：提供清洗好的基准数据集（HumanEval、CodeSearchNet、Python-Doc、Py150），并支持基于编译器（如 LLVM）的特征提取脚本。
    - **支持大规模代码模型**：内置 Code Llama、CodeT5、CodeGen、StarCoder。
    - **基准测试与评估**：统一评测多个下游任务，支持 pass@k 等常用指标。
    - **高效优化**：支持 `torch.distributed` + NCCL 的分布式训练，支持 FP32/FP16 混合精度。
    - **增强日志**：提供清晰、详细的训练与调试日志，便于性能优化。

    ---
    '''
)

# PIPELINE_HELP = dedent(
#     """
#     上传本地图片 **或** 粘贴 URL 以展示项目流程图。
#     - 支持格式：PNG / JPG / GIF / SVG (静态)
#     - 如果同时提供两者，**URL 优先**。
#     """
# )

# ---------------------- Helpers ----------------------

def choose_pipeline_image(url: str, img):
    """优先级：URL > 本地上传 > 无"""
    if url and url.strip():
        return url.strip()
    return img

YOUTUBE_EMBED_TPL = (
    '<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:12px">'
    '<iframe src="https://www.youtube.com/embed/{vid}" '
    'title="NaturalCC 演示视频" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
    'allowfullscreen style="position:absolute;top:0;left:0;width:100%;height:100%"></iframe>'
    '</div>'
)

def to_youtube_embed(url_or_id: str):
    if not url_or_id:
        return ""
    s = url_or_id.strip()
    if "youtube.com" in s or "youtu.be" in s:
        import urllib.parse as up
        try:
            if "v=" in s:
                q = up.urlparse(s)
                vid = up.parse_qs(q.query).get("v", [""])[0]
            else:
                vid = s.split("/")[-1]
        except Exception:
            vid = s
    else:
        vid = s
    vid = vid.split("?")[0]
    return YOUTUBE_EMBED_TPL.format(vid=vid)

# ---------------------- Tutorial ----------------------

TUTORIAL_MD = dedent(
    r'''
    ## 🛠️ 教程

    ### 🔧 安装指南
    请确保系统满足以下要求：
    - GCC/G++ 版本 ≥ 5.0  
    - NVIDIA GPU、NCCL 和 CUDA Toolkit（可选但推荐）  
    - NVIDIA Apex 库（可选，加速训练）  

    **步骤：**
    ```bash
    # (可选) 创建 conda 环境
    conda create -n naturalcc python=3.6
    conda activate naturalcc

    # 从源码构建
    git clone https://github.com/CGCL-codes/naturalcc && cd naturalcc
    pip install -r requirements.txt
    cd src
    pip install --editable ./

    # 安装额外依赖
    conda install conda-forge::libsndfile
    pip install -q -U git+https://github.com/huggingface/transformers.git
    pip install -q -U git+https://github.com/huggingface/accelerate.git
    ```

    **HuggingFace Token**
    某些模型（如 StarCoder）需要 HuggingFace token：
    ```bash
    huggingface-cli login
    ```

    ---

    ### 🚀 快速开始

    #### 示例 1：代码生成 (Code Generation)
    1. 下载模型权重 (如 Codellama-7B)  
    2. 准备测试数据集 (JSON 格式)：  
    ```json
    [
      {"input": "this is a"},
      {"input": "from tqdm import"},
      {"input": "def calculate("},
      {"input": "a = b**2"},
      {"input": "torch.randint"},
      {"input": "x = [1,2"}
    ]
    ```
    3. 运行生成脚本：
    ```python
    print('Initializing GenerationTask')
    task = GenerationTask(task_name="codellama_7b_code", device="cuda:0")

    print('Loading model weights [{}]'.format(ckpt_path))
    task.from_pretrained(ckpt_path)

    print('Processing dataset [{}]'.format(dataset_path))
    task.load_dataset(dataset_path)

    task.run(output_path=output_path, batch_size=1, max_length=50)
    print('Output file: {}'.format(output_path))
    ```

    #### 示例 2：代码摘要 (Code Summarization)
    下载并处理数据集：
    ```bash
    # 下载数据
    bash dataset/python_wan/download.sh
    # 清理
    python -m dataset.python_wan.clean
    # 属性转文件
    python -m dataset.python_wan.attributes_cast

    # 保存 tokens
    python -m dataset.python_wan.summarization.preprocess
    ```

    训练与推理：
    ```bash
    # 训练
    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.transformer.train -f config/python_wan/python > run/summarization/transformer/config/python_wan/python.log 2>&1 &

    # 推理
    CUDA_VISIBLE_DEVICES=0 python -m run.summarization.transformer.eval -f config/python_wan/python -o run/summarization/transformer/config/python_wan/python.txt
    ```

    ---

    ### 📚 数据集支持
    NaturalCC 支持多样化的数据集，包括：
    - Python (Wan et al.)  
    - CodeSearchNet (Husain et al.)  
    - CodeXGlue (Feng et al.)  
    - Py150 (官方/原始)  
    - OpenCL (Grewe et al.)  
    - Java (Hu et al.)  
    - Stack Overflow  
    - DeepCS (Gu et al.)  
    - AVATAR (Ahmad et al.)  
    - StackOverflow (Iyer et al.)  
    '''
)

# ---------------------- UI ----------------------

custom_css = dedent(
    """
    .nc-hero {
        display:flex; gap:18px; align-items:center; margin: 8px 0 18px 0;
        padding: 16px; border-radius: 16px; border: 1px solid var(--block-border-color);
        background: linear-gradient(135deg, rgba(99,102,241,.08), rgba(236,72,153,.06));
        justify-content:center;
        text-align:center;
    }
    .nc-badge {font-weight:600; padding:4px 10px; border-radius:999px; border:1px solid #ddd;}
    .nc-footer {opacity:.8; font-size:.9rem; text-align:center;}
    .gradio-container {max-width: 1080px !important; margin:auto;}
    .gr-block, .gr-row, .gr-column {justify-content:center; text-align:center;}
    """
)

with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.indigo),
               css=custom_css, title=APP_TITLE) as demo:
    gr.HTML(f"""
    <div class='nc-hero'>
        <div>
            <!-- <div class='nc-badge'>NaturalCC — Natural Code Comprehension</div> -->
            <h1 style='margin:.2rem 0'>{APP_TITLE}</h1>
            <p style='margin:0;line-height:1.6'>连接编程语言与自然语言的模块化、可扩展代码智能工具包。</p>
        </div>
    </div>
    """)

    with gr.Tabs():
        with gr.TabItem("🏠 首页"):
            gr.Markdown(APP_DESC)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📦 功能总览") 
                    gr.Image(
                        value="图片测试.png",  # 替换为图片的实际路径或 URL
                        label=None,
                        show_label=False,
                        interactive=False,
                        elem_id="large-image",
                    )
            #         pipeline_img = gr.Image(type="filepath", label="上传流程图", interactive=True)
            #         pipeline_url = gr.Textbox(label="或粘贴图片 URL", placeholder="https://your.cdn/pipeline.png")
            #         gr.Markdown(PIPELINE_HELP)
            #         show_btn = gr.Button("显示流程图", variant="primary")
            #     with gr.Column(scale=1):
            #         pipeline_preview = gr.Image(label="预览", interactive=False)
            # show_btn.click(fn=choose_pipeline_image, inputs=[pipeline_url, pipeline_img], outputs=pipeline_preview)

            with gr.Accordion("ℹ️ 项目说明", open=False):
                gr.Markdown(dedent('''
                - 本应用为项目静态展示界面。
                - 请将教程命令和脚本替换为你仓库的实际路径。
                - 可以新增更多标签页（如 **Playground**）来接入在线模型。
                '''))

        with gr.TabItem("🎬 演示视频"):
            gr.Markdown("#### 输入 YouTube 链接或视频 ID，或上传本地视频文件。")
            with gr.Row():
                with gr.Column():
                    yt_input = gr.Textbox(label="YouTube 链接或视频 ID", placeholder="https://youtu.be/XXXX 或 dQw4w9WgXcQ")
                    yt_btn = gr.Button("嵌入 YouTube 视频")
                    yt_html = gr.HTML()
                with gr.Column():
                    video_upl = gr.Video(label="上传本地演示视频 (mp4/webm)")
            yt_btn.click(fn=to_youtube_embed, inputs=yt_input, outputs=yt_html)
            
            # --- 本地视频演示 ---
            gr.Markdown("#### fun1 演示视频")
            gr.Video(value="视频演示1.mp4", label=None, show_label=False, interactive=False, loop=True, autoplay=True)
            gr.Markdown("一些描述。。。")
            gr.Markdown("---")
            gr.Markdown("#### fun2 演示视频")
            gr.Video(value="视频演示2.mp4", label=None, show_label=False, interactive=False, loop=True, autoplay=True)

        with gr.TabItem("📘 教程"):
            with gr.TabItem("🚀 快速使用"):
                gr.Markdown(TUTORIAL_MD)
            with gr.TabItem("fun1"):
                gr.Markdown("xxx")
            with gr.TabItem("fun2"):
                gr.Markdown("xxx")
            with gr.TabItem("fun3"):
                gr.Markdown("xxx")
            with gr.TabItem("fun4"):
                gr.Markdown("xxx")

    gr.Markdown(dedent('''
    ---
    <div class="nc-footer">由 ❤️ Gradio 构建。
    </div>
    '''))

if __name__ == "__main__":
    demo.launch()
