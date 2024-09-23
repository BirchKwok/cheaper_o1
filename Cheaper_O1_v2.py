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
    return f"""你是一个名为"思考者"（Thinker）的AI助手，正在解决或继续解决问题的过程。请使用与用户提问语言相同的语言回答，或者直接使用用户指定的语言。
当前时间：{current_time}"""

@dataclass
class Step:
    content: str
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


class ProblemSolver:
    def __init__(self):
        self.last_fa_output = None

    def solve_problem_with_history(self, problem: str, history: Optional[str] = None, max_attempts: int = 5) -> ReasoningProcess:
        if history is None:
            history = self.last_fa_output

        bilingual_display("Problem", "问题")
        display(Markdown(f"{problem}"))
        reasoning_process = ReasoningProcess(initial_problem=problem)
        web_info = web_viewer(problem)

        reasoning_process.ia_analysis, steps = self.initial_analysis(problem, web_info, history)
        display_collapsible("Initial Analysis / 初步分析", reasoning_process.ia_analysis)

        for attempt in range(max_attempts):
            reasoning_process.steps = steps

            # 执行每个步骤的T模型
            for i, step in enumerate(reasoning_process.steps):
                step.result = self.execute_step(problem, step, web_info, i+1)
                display_collapsible(f"Step {i+1} Result (Attempt {attempt+1}) / 步骤 {i+1} 结果 (尝试 {attempt+1}) : {step.content}", step.result)

            # 汇总所有步骤的输出结果的T模型
            reasoning_process.final_reasoning = self.summarize_results(problem, reasoning_process.steps)
            display_collapsible(f"Final Reasoning (Attempt {attempt+1}) / 最终推理 (尝试 {attempt+1})", reasoning_process.final_reasoning)

            # 判断模型：核对推理结果是否正确
            reasoning_process.judgment = self.judge_result(problem, reasoning_process.ia_analysis, reasoning_process.steps, reasoning_process.final_reasoning, web_info)
            display_collapsible(f"Judgment (Attempt {attempt+1}) / 判断 (尝试 {attempt+1})", reasoning_process.judgment)

            if self.is_result_correct(reasoning_process.judgment):
                break

        # FA模型：整合答案
        reasoning_process.final_answer = self.final_answer_fa(problem, reasoning_process.final_reasoning, reasoning_process.judgment)

        bilingual_display("Final Answer", "最终答案")
        display(Markdown(f"{reasoning_process.final_answer}"))

        # 更新last_fa_output
        self.last_fa_output = reasoning_process.final_answer

        return reasoning_process

    def initial_analysis(self, problem: str, web_info: str, history: Optional[str]) -> Tuple[str, List[Step]]:
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
4. 如果不需要网络搜索信息，请忽略。
5. 请严格按照以下XML格式返回结果：

<analysis>
    <step>
        <content>步骤1的内容</content>
        <description>对执行这个步骤的模型的简短指导</description>
    </step>
    <step>
        <content>步骤2的内容</content>
        <description>对执行这个步骤的模型的简短指导</description>
    </step>
    <step>
        <content>步骤3的内容</content>
        <description>对执行这个步骤的模型的简短指导</description>
    </step>
</analysis>
        """
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        if history:
            messages.insert(1, {"role": "assistant", "content": history})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3
        ).choices[0].message.content.strip()

        steps = self.parse_steps_xml(response)
        return response, steps

    def parse_steps_xml(self, analysis: str) -> List[Step]:
        try:
            root = ET.fromstring(analysis)
            steps = []
            for step_elem in root.findall('.//step'):
                content = step_elem.find('content').text.strip() if step_elem.find('content') is not None else ""
                description = step_elem.find('description').text.strip() if step_elem.find('description') is not None else ""
                steps.append(Step(content=content, description=description))
            return steps
        except ET.ParseError as e:
            print("XML解析错误:", e)
            return self.parse_steps_fallback(analysis)

    def parse_steps_fallback(self, analysis: str) -> List[Step]:
        steps = re.findall(r'<step>\s*<content>(.*?)</content>\s*<description>(.*?)</description>\s*</step>', analysis, re.DOTALL)
        if not steps:
            # 如果无法匹配新格式，尝试匹配旧格式
            old_steps = re.findall(r'^\d+\.\s*(.+)', analysis, re.MULTILINE)
            return [Step(content=step, description="") for step in old_steps]
        return [Step(content=content.strip(), description=description.strip()) for content, description in steps]

    def execute_step(self, problem: str, step: Step, web_info: str, step_number: int) -> str:
        prompt = f"""
根据以下信息，执行任务流程并完成步骤 {step_number}：

问题：{problem}

步骤内容：{step.content}

步骤指导：{step.description}

网络搜索信息：{web_info}

请详细完成上述步骤，并给出相应的结果。
        """
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7  # 统一设定温度
        ).choices[0].message.content.strip()
        return response

    def summarize_results(self, problem: str, steps: List[Step]) -> str:
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
            temperature=0.7
        ).choices[0].message.content.strip()
        return response

    def judge_result(self, problem: str, ia_analysis: str, steps: List[Step], final_reasoning: str, web_info: str) -> str:
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
            temperature=0.7
        ).choices[0].message.content.strip()
        return response

    def is_result_correct(self, judgment: str) -> bool:
        return "结果正确" in judgment

    def final_answer_fa(self, problem: str, final_reasoning: str, judgment: str) -> str:
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
            temperature=0.7
        ).choices[0].message.content.strip()
        return response

# 使用示例
# solver = ProblemSolver()
# result = solver.solve_problem_with_history("你的问题")
