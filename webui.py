import gradio as gr
import re
import xml.etree.ElementTree as ET
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
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


def get_current_time():
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")

def get_system_prompt():
    current_time = get_current_time()
    return f"""你是一个名为"思考者"（Thinker）的AI助手，非常擅长解决复杂的STEM问题。请使用与用户提问语言相同的语言回答，或者直接使用用户指定的语言。
当前时间：{current_time}"""

@dataclass
class Step:
    content: str
    description: str
    result: str = ""

@dataclass
class Attempt:
    number: int
    steps: List[Step]
    final_reasoning: str
    judgment: str

@dataclass
class ReasoningProcess:
    initial_problem: str
    ia_analysis: str = ""
    attempts: List[Attempt] = field(default_factory=list)
    final_answer: Optional[str] = None

def split_description(description: str) -> str:
    description = re.sub(r"步骤\d+：", "", description.strip())
    description = re.sub(r'步骤\d+： ', '', description.strip())
    return description


class ProblemSolver:
    def __init__(self):
        self.last_fa_output = None

    def solve_problem_with_history(self, problem: str, history: Optional[str] = None, max_attempts: int = 5):
        if history is None:
            history = self.last_fa_output

        reasoning_process = ReasoningProcess(initial_problem=problem)
        web_info = web_viewer(problem)

        yield f"思考中...\n正在进行初步分析"
        reasoning_process.ia_analysis, initial_steps = self.initial_analysis(problem, web_info, history)
        yield f"初步分析完成。\n开始尝试解决问题..."

        for attempt_number in range(1, max_attempts + 1):
            steps = [Step(content=step.content, description=step.description) for step in initial_steps]
            yield f"正在进行第 {attempt_number} 次尝试"

            for i, step in enumerate(steps):
                yield f"正在进行第 {attempt_number} 次尝试 (步骤 {i+1}/{len(steps)})"
                step.result = self.execute_step(problem, step, web_info, i+1)

            yield f"第 {attempt_number} 次尝试完成，正在总结结果"
            final_reasoning = self.summarize_results(problem, steps)

            yield f"正在判断第 {attempt_number} 次尝试的结果"
            judgment = self.judge_result(problem, reasoning_process.ia_analysis, steps, final_reasoning, web_info)

            attempt = Attempt(number=attempt_number, steps=steps, final_reasoning=final_reasoning, judgment=judgment)
            reasoning_process.attempts.append(attempt)

            if self.is_result_correct(judgment):
                yield "找到正确答案，正在生成最终回答"
                break

        reasoning_process.final_answer = self.final_answer_fa(problem, reasoning_process.attempts[-1].final_reasoning, reasoning_process.attempts[-1].judgment)
        self.last_fa_output = reasoning_process.final_answer

        yield reasoning_process

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

        if history and isinstance(history, str):
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

请结合网络搜索信息（如果不需要网络搜索信息就可以解决，请忽略），详细完成上述步骤，并给出相应的结果。


注意:
1. 只能完成步骤内容和步骤指导中的要求，不可以尝试在这一个步骤中直接解答问题
2. 反思总结，如果当前步骤没有完成，请说明原因，并提出改进措施。如果当前步骤已经完成，请给出详细结果。
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


注意：
1. 答案只能从给定的信息中推理得出，不可以使用网络搜索信息
2. 答案必须简洁明了，不可以有多余的描述
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

注意：
1. 答案只能从给定的信息中推理得出，不可以使用网络搜索信息
2. 答案必须简洁明了，不可以有多余的描述

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

solver = ProblemSolver()

def user_input(user_message, history):
    return "", history + [[user_message, None]]

def chat(history):
    history_copy = history.copy()
    message = history_copy[-1][0]
    history_copy[-1][1] = "思考中..."
    yield history_copy

    # 提取最后一次助手的回复（排除当前正在处理的消息）
    last_assistant_message = None
    if history_copy[:-1]:
        for user_msg, assistant_msg in reversed(history_copy[:-1]):
            if assistant_msg is not None:
                last_assistant_message = assistant_msg
                break

    for update in solver.solve_problem_with_history(message, last_assistant_message):
        if isinstance(update, str):
            history_copy[-1][1] = update
            yield history_copy
        elif isinstance(update, ReasoningProcess):
            # 构建折叠内容
            attempts_content = ""
            for attempt in update.attempts:
                attempts_content += f"""
<details>
    <summary>尝试 {attempt.number}</summary>
    {''.join([f"<p><strong>步骤 {i+1}：</strong>{step.content}<br><strong>结果：</strong>{step.result}</p>" for i, step in enumerate(attempt.steps)])}
    <p><strong>本次推理：</strong>{attempt.final_reasoning}</p>
    <p><strong>判断：</strong>{attempt.judgment}</p>
</details>
                """

            collapsible_content = f"""
<details>
    <summary>思考过程</summary>
    <p><strong>初步分析：</strong>{update.ia_analysis}</p>
    {attempts_content}
</details>
            """

            full_response = collapsible_content + update.final_answer
            history_copy[-1][1] = full_response
            yield history_copy

def launch_ui():
    with gr.Blocks(title="Cheaper (and slower) O1 simulation model") as demo:
        gr.HTML("<h1 style='text-align: center;'>Cheaper (and slower) O1 simulation model</h1>")
        chatbot = gr.Chatbot()
        with gr.Row():
            msg = gr.Textbox(show_label=False, placeholder="输入您的问题...", scale=8)
            send_button = gr.Button("Post/Stop", scale=1)

        clear = gr.Button("清除对话")

        msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
            chat, [chatbot], [chatbot]
        )
        send_button.click(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
            chat, [chatbot], [chatbot]
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch()

if __name__ == "__main__":
    launch_ui()
