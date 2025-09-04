# 导入 Gradio 库，通常简写为 gr
import gradio as gr
# 导入 textwrap 的 dedent 函数，用于处理多行字符串的缩进，使代码更美观
from textwrap import dedent

# --- 1. 定义全局变量和辅助函数 ---

# 定义应用的标题，方便后续复用
APP_TITLE = "NaturalCC — Natural Code Comprehension"

# 使用 dedent 定义应用的多行描述文本。
# r'''...''' 表示这是一个原始多行字符串，可以包含特殊字符而无需转义。
# Markdown 语法（如 #, **, ---）将由 Gradio 的 Markdown 组件渲染。
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

# ---------------------- 辅助函数 (Helpers) ----------------------

def choose_pipeline_image(url: str, img):
    """
    一个简单的辅助函数，用于决定显示哪个图片。
    逻辑是：如果 URL 输入框不为空，则优先使用 URL；否则使用用户上传的图片。
    """
    if url and url.strip():
        return url.strip()
    return img

# 定义一个 YouTube 视频嵌入的 HTML 模板。
# {vid} 是一个占位符，后续会被替换为实际的 YouTube 视频 ID。
# 这段 HTML 代码创建了一个响应式的容器，让视频可以自适应宽高。
YOUTUBE_EMBED_TPL = (
    '<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:12px">'
    '<iframe src="https://www.youtube.com/embed/{vid}" '
    'title="NaturalCC 演示视频" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
    'allowfullscreen style="position:absolute;top:0;left:0;width:100%;height:100%"></iframe>'
    '</div>'
)

def to_youtube_embed(url_or_id: str):
    """
    这个函数负责将用户输入的 YouTube 链接或视频 ID 转换成可嵌入的 HTML 代码。
    它会解析不同格式的 YouTube 链接，提取出视频 ID，然后填入上面的 HTML 模板中。
    """
    if not url_or_id:
        return ""
    s = url_or_id.strip()
    # 检查输入是否是 YouTube 链接
    if "youtube.com" in s or "youtu.be" in s:
        import urllib.parse as up
        try:
            # 解析标准链接 (e.g., ...?v=VIDEO_ID)
            if "v=" in s:
                q = up.urlparse(s)
                vid = up.parse_qs(q.query).get("v", [""])[0]
            # 解析短链接 (e.g., youtu.be/VIDEO_ID)
            else:
                vid = s.split("/")[-1]
        except Exception:
            vid = s
    # 如果不是链接，就假定输入的是视频 ID
    else:
        vid = s
    # 去掉链接中可能存在的其他参数 (e.g., ?t=120s)
    vid = vid.split("?")[0]
    # 将视频 ID 填入模板并返回最终的 HTML
    return YOUTUBE_EMBED_TPL.format(vid=vid)

# ---------------------- 教程内容 ----------------------

# 定义教程页面的 Markdown 内容
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
    git clone [https://github.com/CGCL-codes/naturalcc](https://github.com/CGCL-codes/naturalcc) && cd naturalcc
    pip install -r requirements.txt
    cd src
    pip install --editable ./

    # 安装额外依赖
    conda install conda-forge::libsndfile
    pip install -q -U git+[https://github.com/huggingface/transformers.git](https://github.com/huggingface/transformers.git)
    pip install -q -U git+[https://github.com/huggingface/accelerate.git](https://github.com/huggingface/accelerate.git)
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

# ---------------------- 3. UI 界面布局 ----------------------

# 定义自定义 CSS 样式，用于美化界面
custom_css = dedent(
    """
    /* 自定义英雄区域（顶部标题栏）样式 */
    .nc-hero {
        display:flex; gap:18px; align-items:center; margin: 8px 0 18px 0;
        padding: 16px; border-radius: 16px; border: 1px solid var(--block-border-color);
        background: linear-gradient(135deg, rgba(99,102,241,.08), rgba(236,72,153,.06));
        justify-content:center;
        text-align:center;
    }
    .nc-badge {font-weight:600; padding:4px 10px; border-radius:999px; border:1px solid #ddd;}
    /* 页脚样式 */
    .nc-footer {opacity:.8; font-size:.9rem; text-align:center;}
    /* Gradio 容器最大宽度，使其居中显示 */
    .gradio-container {max-width: 1080px !important; margin:auto;}
    /* 块、行、列内容居中 */
    .gr-block, .gr-row, .gr-column {justify-content:center; text-align:center;}
    """
)

# 【核心语法】: gr.Blocks() 是 Gradio 中功能更强大的界面构建方式，可以自由组合各种组件。
# 它就像一个画板，你可以在上面放置各种元素。
# with gr.Blocks(...) as demo: 表示创建一个 Blocks 实例，并将其赋值给 demo。
# 后续所有在这个 `with` 语句块中创建的 Gradio 组件都会被自动添加到这个画板上。
# theme: 设置界面主题，这里使用了 Soft 主题，并指定了主色调。
# css: 引入上面定义的自定义 CSS。
# title: 设置浏览器标签页的标题。
with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.indigo),
               css=custom_css, title=APP_TITLE) as demo:
    
    # 【组件】: gr.HTML() 可以直接在界面上渲染 HTML 代码。
    # 这里用它来创建页面的顶部标题栏（Hero Section）。
    gr.HTML(f"""
    <div class='nc-hero'>
        <div>
            <!-- <div class='nc-badge'>NaturalCC — Natural Code Comprehension</div> -->
            <h1 style='margin:.2rem 0'>{APP_TITLE}</h1>
            <p style='margin:0;line-height:1.6'>连接编程语言与自然语言的模块化、可扩展代码智能工具包。</p>
        </div>
    </div>
    """)

    # 【布局】: gr.Tabs() 创建一个选项卡布局。
    # with gr.Tabs() as tabs: 之后的所有 gr.TabItem 都会成为这个选项卡组的一部分。
    with gr.Tabs() as tabs:
        # 【布局】: gr.TabItem("标题") 创建一个具体的选项卡。
        # with gr.TabItem(...): 内部的组件将显示在这个选项卡页面中。
        with gr.TabItem("🏠 首页"):
            # 【组件】: gr.Markdown() 用于显示 Markdown 格式的文本。
            gr.Markdown(APP_DESC)
            
            # 【布局】: gr.Row() 创建一个水平行，内部的组件会横向排列。
            with gr.Row():
                # 【布局】: gr.Column() 创建一个垂直列。
                # scale=1: 在 Row 中，可以设置 scale 来控制列的宽度比例。
                with gr.Column(scale=1):
                    gr.Markdown("### 📦 功能总览")
                    # 【组件】: gr.Image() 用于显示图片。
                    gr.Image(
                        value="https://placehold.co/800x400/a2d2ff/ffffff?text=功能总览流程图",  # value: 默认显示的图片路径或 URL。这里用一个占位图代替。
                        label=None,                # label: 组件上方的标签文本，None 表示不显示。
                        show_label=False,          # show_label: 是否显示标签。
                        interactive=False,         # interactive: 用户是否可以交互（如上传图片），False 表示仅用于显示。
                        elem_id="large-image",     # elem_id: 给这个 HTML 元素指定一个 ID，方便用 CSS 选择。
                    )

            # 【组件】: gr.Accordion("标题", open=False) 创建一个可折叠/展开的部分。
            # open=False: 默认是折叠状态。
            with gr.Accordion("ℹ️ 项目说明", open=False):
                gr.Markdown(dedent('''
                - 本应用为项目静态展示界面。
                - 请将教程命令和脚本替换为你仓库的实际路径。
                - 可以新增更多标签页（如 **Playground**）来接入在线模型。
                '''))

        # --- 如何插入视频的讲解 ---
        # 这里我们创建一个新的选项卡 "演示视频"
        with gr.TabItem("🎬 演示视频"):
            gr.Markdown("#### 输入 YouTube 链接或视频 ID，或上传本地视频文件。")
            with gr.Row():
                # 在左侧列放置 YouTube 嵌入功能
                with gr.Column():
                    # 【组件】: gr.Textbox() 创建一个文本输入框。
                    yt_input = gr.Textbox(label="YouTube 链接或视频 ID", placeholder="https://youtu.be/XXXX 或 dQw4w9WgXcQ")
                    # 【组件】: gr.Button() 创建一个按钮。
                    yt_btn = gr.Button("嵌入 YouTube 视频")
                    # 【组件】: gr.HTML() 创建一个空的 HTML 组件，用于后续显示嵌入的视频。
                    yt_html = gr.HTML()

                # 在右侧列放置本地视频上传功能
                with gr.Column():
                    # 【核心语法】: gr.Video() 是专门用来处理视频的组件。
                    # 当 interactive 为 True (默认) 时，它会显示一个上传区域，用户可以拖拽或点击上传视频文件。
                    # 上传后，视频会直接在这个组件中播放。
                    video_upl = gr.Video(label="上传本地演示视频 (mp4/webm)")
            
            # 【事件处理】: .click(fn, inputs, outputs) 是 Gradio 的核心交互逻辑。
            # 这行代码的意思是：
            # 当 yt_btn (按钮) 被点击 (click) 时:
            # 1. 调用 fn=to_youtube_embed 这个函数。
            # 2. 将 inputs=[yt_input] (文本输入框的内容) 作为函数的参数传入。
            # 3. 将函数的返回值，更新到 outputs=[yt_html] (HTML 组件) 中。
            # 这样就实现了“点击按钮，将YouTube链接转换成HTML并显示出来”的完整流程。
            yt_btn.click(fn=to_youtube_embed, inputs=yt_input, outputs=yt_html)

        with gr.TabItem("📘 教程"):
            # 在一个选项卡内部，还可以嵌套另一个 Tabs，实现多级选项卡
            with gr.Tabs():
                 with gr.TabItem("🚀 快速使用"):
                     gr.Markdown(TUTORIAL_MD)
                 with gr.TabItem("fun1"):
                     gr.Markdown("这里是功能1的教程内容")
                 with gr.TabItem("fun2"):
                     gr.Markdown("这里是功能2的教程内容")
                 with gr.TabItem("fun3"):
                     gr.Markdown("这里是功能3的教程内容")
                 with gr.TabItem("fun4"):
                     gr.Markdown("这里是功能4的教程内容")

    # 在所有选项卡外部，添加一个页脚
    gr.Markdown(dedent('''
    ---
    <div class="nc-footer">由 ❤️ Gradio 构建。
    </div>
    '''))


# --- 4. 启动应用 ---

# 这段代码确保只有在直接运行这个 Python 文件时，才会启动 Gradio 服务。
# 如果这个文件被其他文件导入，则不会执行。
if __name__ == "__main__":
    # demo.launch() 启动 Gradio 应用。
    # 它会启动一个本地 Web 服务器，并生成一个可以访问的 URL。
    demo.launch()
