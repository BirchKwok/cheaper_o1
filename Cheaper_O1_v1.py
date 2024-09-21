import re
import xml.etree.ElementTree as ET
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from IPython.display import display, Markdown, HTML
from webviewer import web_viewer
from datetime import datetime


# 初始化 OpenAI 客户端
with open("api_key", "r") as f:
    api_key = f.read().strip()

client = OpenAI(
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)
model = "glm-4-flash"

display(HTML("""
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
function renderMarkdown() {
    const details = document.querySelectorAll('details');
    details.forEach(detail => {
        const content = detail.querySelector('.markdown-content');
        if (content) {
            content.innerHTML = marked.parse(content.textContent);
        }
    });
}
</script>
"""))

def get_current_time():
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")

def get_system_prompt():
    current_time = get_current_time()
    return f"""你是一个名为"思考者"（Thinker）的AI助手，正在继续解决问题的过程。请使用与用户提问语言相同的语言回答，或者直接使用用户指定的语言。
当前时间：{current_time}"""

@dataclass
class Step:
    description: str
    result: str = ""

@dataclass
class ReasoningProcess:
    initial_problem: str
    ia_analysis: str = ""
    steps: List[Step] = field(default_factory=list)
    final_reasoning: str = ""
    judgment: str = ""
    final_answer: Optional[str] = None

def bilingual_display(text_en, text_zh):
    display(Markdown(f"**{text_en.strip()}** / **{text_zh.strip()}**"))

def display_collapsible(summary, content, is_markdown=True):
    html = f"""
    <details>
        <summary><strong>{summary}</strong></summary>
        <div class="{'markdown-content' if is_markdown else ''}">
{content}
        </div>
    </details>
    <script>
    renderMarkdown();
    </script>
    """
    display(HTML(html))

def split_description(description: str) -> str:
    description = re.sub(r"步骤\d+：", "", description.strip())
    description = re.sub(r'步骤\d+： ', '', description.strip())
    return description


def solve_problem(problem: str, max_attempts: int = 5) -> ReasoningProcess:
    bilingual_display("Problem", "问题")
    display(Markdown(f"{problem}"))
    reasoning_process = ReasoningProcess(initial_problem=problem)
    web_info = web_viewer(problem)

    # IA模型：分析问题和规划任务流程
    reasoning_process.ia_analysis, steps = initial_analysis(problem, web_info)
    display_collapsible("Initial Analysis / 初步分析", reasoning_process.ia_analysis)

    for attempt in range(max_attempts):
        reasoning_process.steps = [Step(description=step) for step in steps]

        # 执行每个步骤的T模型
        for i, step in enumerate(reasoning_process.steps):
            step.result = execute_step(problem, step.description, web_info, i+1)
            display_collapsible(f"Step {i+1} Result (Attempt {attempt+1}) / 步骤 {i+1} 结果 (尝试 {attempt+1}) :  {split_description(step.description)}", step.result)

        # 汇总所有步骤的输出结果的T模型
        reasoning_process.final_reasoning = summarize_results(problem, reasoning_process.steps)
        display_collapsible(f"Final Reasoning (Attempt {attempt+1}) / 最终推理 (尝试 {attempt+1})", reasoning_process.final_reasoning)

        # 判断模型：核对推理结果是否正确
        reasoning_process.judgment = judge_result(problem, reasoning_process.ia_analysis, reasoning_process.steps, reasoning_process.final_reasoning, web_info)
        display_collapsible(f"Judgment (Attempt {attempt+1}) / 判断 (尝试 {attempt+1})", reasoning_process.judgment)

        if is_result_correct(reasoning_process.judgment):
            break

    # FA模型：整合答案
    reasoning_process.final_answer = final_answer_fa(problem, reasoning_process.final_reasoning, reasoning_process.judgment)

    bilingual_display("Final Answer", "最终答案")
    display(Markdown(f"{reasoning_process.final_answer}"))

    return reasoning_process

def initial_analysis(problem: str, web_info: str) -> Tuple[str, List[str]]:
    prompt = f"""
作为一个擅长解决复杂STEM问题的思考者（Thinker），请分析以下问题，思考可能的解决方法，并规划后续解决步骤。
请尽量减少步骤数量，每个步骤应简洁明了且不可再拆分。
不要给出最终答案，只需提供分析和计划。

问题：
{problem}

网络搜索信息：
{web_info}

注意：
1. 请提供你的分析和逐步计划。你的分析应该紧密围绕问题"{problem}"，并且逐步计划应该详细到每个步骤，但尽量保持步骤数量最少。
2. 每个步骤应独立且不重复，避免不必要的重复处理。
3. 请不要给出最终答案，只需提供分析和计划。
4. 请按照以下XML格式返回结果：
<analysis>
    <step>步骤1</step>
    <step>步骤2</step>
    <step>步骤3</step>
</analysis>
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3  # 设置较低的温度以减少随机性
    ).choices[0].message.content.strip()

    # 解析XML格式的步骤
    steps = parse_steps_xml(response)
    return response, steps

def parse_steps_xml(analysis: str) -> List[str]:
    try:
        root = ET.fromstring(analysis)
        steps = [step.text.strip() for step in root.findall('.//step') if step.text]
        # 去除可能的重复步骤
        unique_steps = list(dict.fromkeys(steps))
        return unique_steps
    except ET.ParseError as e:
        print("XML解析错误:", e)
        # 备用策略：尝试使用正则解析
        return parse_steps_fallback(analysis)

def parse_steps_fallback(analysis: str) -> List[str]:
    # 备用的步骤解析方法
    steps = re.findall(r'^\d+\.\s*(.+)', analysis, re.MULTILINE)
    if not steps:
        steps = re.findall(r'^- \s*(.+)', analysis, re.MULTILINE)
    # 去除可能的重复步骤
    unique_steps = list(dict.fromkeys(steps))
    return unique_steps

def execute_step(problem: str, step_description: str, web_info: str, step_number: int) -> str:
    prompt = f"""
根据以下信息，执行任务流程并完成步骤 {step_number}：

问题：{problem}

步骤描述：{step_description}

网络搜索信息：{web_info}

请详细完成上述步骤，并给出相应的结果。
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3  # 统一设定温度
    ).choices[0].message.content.strip()
    return response

def summarize_results(problem: str, steps: List[Step]) -> str:
    combined_results = "\n".join([f"步骤 {i+1} 结果：{step.result}" for i, step in enumerate(steps)])
    prompt = f"""
请综合以下步骤的结果，进行最终的推理：

问题：{problem}

步骤结果：
{combined_results}

请根据这些信息，给出一个综合的推理结论。
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    ).choices[0].message.content.strip()
    return response

def judge_result(problem: str, ia_analysis: str, steps: List[Step], final_reasoning: str, web_info: str) -> str:
    combined_steps = "\n".join([f"步骤 {i+1} 描述：{step.description}\n步骤 {i+1} 结果：{step.result}" for i, step in enumerate(steps)])
    prompt = f"""
请综合以下信息，判断当前的推理结果是否正确，是否需要进一步迭代：

原始问题：{problem}

初步分析：{ia_analysis}

步骤及结果：
{combined_steps}

最终推理：
{final_reasoning}

网络搜索信息：
{web_info}

请给出你的判断，如果认为答案已经正确，请明确说明"结果正确"，并简要总结正确的答案。如果认为还需要进一步迭代，请说明原因。
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    ).choices[0].message.content.strip()
    return response

def is_result_correct(judgment: str) -> bool:
    return "结果正确" in judgment

def final_answer_fa(problem: str, final_reasoning: str, judgment: str) -> str:
    prompt = f"""
请根据以下信息，整合出一个简洁的最终答案：

原始问题：{problem}

最终推理：{final_reasoning}

判断结果：{judgment}

请给出一个简洁的最终答案，确保答案与原始问题直接相关。
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    ).choices[0].message.content.strip()
    return response
