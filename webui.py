import gradio as gr
import re
import xml.etree.ElementTree as ET
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from webviewer import web_viewer
from datetime import datetime


# 初始化 OpenAI 客户端
with open("api_key", "r") as f:
    api_key = f.read().strip()

client = OpenAI(
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)
model = "glm-4-plus"

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

class PromptTemplate:
    def __init__(self, template, tags):
        self.template = template
        self.tags = tags
        self.performance_score = 1.0
        self.use_count = 0


class PromptOptimizer:
    def __init__(self):
        self.prompt_templates = [
            PromptTemplate("作为一个擅长解决复杂{problem_type}问题的思考者（Thinker），请分析以下问题，思考可能的解决方法，并规划后续解决步骤。", ["general"]),
            PromptTemplate("你是一位{field}领域的专家。请详细分析这个{problem_type}问题，并提供一个系统的解决方案。", ["expert"]),
            PromptTemplate("假设你正在向一个{level}学生解释这个{problem_type}问题。请提供一个清晰、易懂的解决方案。", ["educational"]),
            PromptTemplate("在{field}研究中，我们经常遇到类似的{problem_type}问题。请详细说明你会如何系统地解决这个问题。", ["research"]),
            PromptTemplate("作为一个跨学科专家，你如何将{field}的知识应用到解决这个{problem_type}问题？请提供详细的步骤。", ["interdisciplinary"]),
            PromptTemplate("在实际应用中，这个{problem_type}问题可能会如何出现？请提供一个实用的解决方案。", ["practical"]),
            PromptTemplate("如果你要设计一个算法来解决这个{problem_type}问题，你会如何着手？请提供详细的思路。", ["algorithmic"]),
            PromptTemplate("从理论和实践的角度，这个{problem_type}问题有哪些可能的解决方案？请进行对比分析。", ["theoretical-practical"]),
            PromptTemplate("如果资源有限，你会如何优化解决这个{problem_type}问题的方法？请提供一个高效的解决方案。", ["optimization"]),
            PromptTemplate("这个{problem_type}问题可能涉及哪些潜在的陷阱或误区？请在提供解决方案的同时，指出需要注意的关键点。", ["critical-thinking"]),
            PromptTemplate("作为一名{field}领域的创新者，你如何用创新的方法来解决这个{problem_type}问题？请提出一些非传统的解决方案。", ["innovative"]),
            PromptTemplate("假设你是一位{field}历史学家，这个{problem_type}问题在历史上是如何被解决的？请提供历史视角和现代应用。", ["historical"]),
            PromptTemplate("作为一名{field}领域的质量保证专家，你会如何确保这个{problem_type}问题的解决方案是可靠和高质量的？请提供详细的质量控制步骤。", ["quality-assurance"]),
            PromptTemplate("你是一位{field}领域的未来学家，考虑到技术发展趋势，这个{problem_type}问题在未来可能如何演变？请提供前瞻性的解决方案。", ["futuristic"]),
            PromptTemplate("作为一名{field}领域的伦理学家，这个{problem_type}问题可能涉及哪些伦理考量？请在提供解决方案时考虑伦理影响。", ["ethical"]),
            PromptTemplate("假设你是一位{field}领域的项目经理，你会如何组织团队来解决这个{problem_type}问题？请提供项目规划和资源分配建议��", ["project-management"]),
            PromptTemplate("作为一名{field}领域的风险分析师，这个{problem_type}问题可能带来哪些潜在风险？请在解决方案中包含风险评估和缓解策略。", ["risk-analysis"]),
            PromptTemplate("你是一位{field}领域的可持续发展专家，如何以可持续的方式解决这个{problem_type}问题？请考虑长期影响和环境因素。", ["sustainability"]),
            PromptTemplate("作为一名{field}领域的用户体验设计师，你如何确保这个{problem_type}问题的解决方案是用户友好的？请提供以用户为中心的解决方案。", ["user-experience"]),
            PromptTemplate("假设你是一位{field}领域的政策制定者，你会提出什么样的政策或法规来解决这个{problem_type}问题？请考虑社会和经济影响。", ["policy-making"]),
            PromptTemplate("作为一名{field}领域的数据科学家，你会如何运用数据分析和机器学习技术来解决这个{problem_type}问题？请提供基于数据的解决方案。", ["data-science"]),
            PromptTemplate("假设你是一位{field}领域的营销专家，如何将解决这个{problem_type}问题的方案有效地传达给目标受众？请提供营销策略。", ["marketing"]),
            PromptTemplate("作为一名{field}领域的人力资源管理者，这个{problem_type}问题可能如何影响员工和组织文化？请提供人才管理的角度。", ["human-resources"]),
            PromptTemplate("你是一位{field}领域的财务分析师，如何评估解决这个{problem_type}问题的财务影响？请提供成本效益分析。", ["financial-analysis"]),
            PromptTemplate("作为一名{field}领域的法律顾问，这个{problem_type}问题可能涉及哪些法律风险？请提供法律合规的解决方案。", ["legal"]),
            PromptTemplate("假设你是一位{field}领域的供应链专家，如何优化解决这个{problem_type}问题的供应链流程？请考虑全球化因素。", ["supply-chain"]),
            PromptTemplate("作为一名{field}领域的环境科学家，这个{problem_type}问题对环境有何影响？请提供环保的解决方案。", ["environmental-science"]),
            PromptTemplate("你是一位{field}领域的社会学家，这个{problem_type}问题如何影响社会结构和群体互动？请从社会学角度分析。", ["sociology"]),
            PromptTemplate("作为一名{field}领域的心理学家，这个{problem_type}问题可能对个人或群体心理有何影响？请提供心理学视角的解决方案。", ["psychology"]),
            PromptTemplate("假设你是一位{field}领域的教育工作者，如何将这个{problem_type}问题的解决过程转化为教育资源？请设计相关的教学方案。", ["education"]),
            PromptTemplate("作为一名{field}领域的艺术家或设计师，你会如何用创意方式表达或解决这个{problem_type}问题？请提供艺术或设计方案。", ["art-design"]),
            PromptTemplate("你是一位{field}领域的医疗保健专业人士，这个{problem_type}问题可能如何影响公共卫生？请提供医疗卫生角度的解决方案。", ["healthcare"]),
            PromptTemplate("作为一名{field}领域的城市规划师，这个{problem_type}问题如何影响城市发展？请提供城市规划的视角和解决方案。", ["urban-planning"]),
            PromptTemplate("假设你是一位{field}领域的农业专家，如何将这个{problem_type}问题应用到现代农业中？请考虑可持续农业实践。", ["agriculture"]),
            PromptTemplate("作为一名{field}领域的能源专家，这个{problem_type}问题与能源利用有何关联？请提供节能和可再生能源的解决方案。", ["energy"]),
        ]
        self.problem_vectors = []
        self.problem_templates = []

    def choose_prompt(self, problem, messages):
        analysis = self.analyze_problem(problem)

        formatted_templates = [template.template.format(**analysis) for template in self.prompt_templates]

        prompt = f"""
给定以下问题：

{problem}

以及以下可能的prompt模板：

{formatted_templates}

请选择最适合解决这个问题的prompt模板。你的回答应该只包含所选模板的编号（从1开始）。例如，如果你认为第3个模板最合适，就回答"3"。
        """

        response = client.chat.completions.create(
            model=model,
            messages=messages + [{"role": "user", "content": prompt}],
            temperature=0.3
        ).choices[0].message.content.strip()

        try:
            chosen_index = int(response) - 1
            chosen_template = self.prompt_templates[chosen_index]
        except (ValueError, IndexError):
            # 如果大模型的回答无效，就使用默认的选择方法
            chosen_template = max(self.prompt_templates, key=lambda t: t.performance_score)

        chosen_template.use_count += 1
        return chosen_template.template.format(**analysis)

    def update(self, prompt, reward):
        for template in self.prompt_templates:
            if template.template in prompt:
                template.performance_score = (template.performance_score * template.use_count + reward) / (template.use_count + 1)
                break


    def analyze_problem(self, problem):
        problem_lower = problem.lower()

        # 数学类问题
        if any(keyword in problem_lower for keyword in ["数学", "方程", "几何", "代数", "计算", "证明"]):
            return {"problem_type": "数学", "field": "数学", "level": "高中"}

        # 物理类问题
        elif any(keyword in problem_lower for keyword in ["物理", "力学", "电磁学", "热力学", "光学", "量子"]):
            return {"problem_type": "物理", "field": "物理学", "level": "大学"}

        # 化学类问题
        elif any(keyword in problem_lower for keyword in ["化学", "反应", "分子", "元素", "化合物", "酸碱"]):
            return {"problem_type": "化学", "field": "化学", "level": "高中"}

        # 生物类问题
        elif any(keyword in problem_lower for keyword in ["生物", "生态", "遗传", "细胞", "进化", "基因"]):
            return {"problem_type": "生物", "field": "生物学", "level": "大学"}

        # 计算机科学类问题
        elif any(keyword in problem_lower for keyword in ["编程", "算法", "数据结构", "计算机", "软件", "网络"]):
            return {"problem_type": "计算机科学", "field": "计算机科学", "level": "大学"}

        # 工程类问题
        elif any(keyword in problem_lower for keyword in ["工程", "机械", "电路", "结构", "设计", "制造"]):
            return {"problem_type": "工程", "field": "工程学", "level": "大学"}

        # 经济金融类问题
        elif any(keyword in problem_lower for keyword in ["经济", "金融", "投资", "市场", "股票", "货币"]):
            return {"problem_type": "经济金融", "field": "经济学", "level": "大学"}

        # 历史类问题
        elif any(keyword in problem_lower for keyword in ["历史", "朝代", "事件", "人物", "文化", "考古"]):
            return {"problem_type": "历史", "field": "历史学", "level": "高中"}

        # 文学类问题
        elif any(keyword in problem_lower for keyword in ["文学", "诗歌", "小说", "作家", "文章", "写作"]):
            return {"problem_type": "文学", "field": "文学", "level": "高中"}

        # 哲学类问题
        elif any(keyword in problem_lower for keyword in ["哲学", "思想", "逻辑", "伦理", "存在", "认知"]):
            return {"problem_type": "哲学", "field": "哲学", "level": "大学"}

        # 如果无法确定具体类型，返回通用类型
        else:
            return {"problem_type": "一般", "field": "综合", "level": "成人"}

class ProblemSolver:
    def __init__(self):
        self.last_fa_output = None
        self.prompt_optimizer = PromptOptimizer()

    def solve_problem_with_history(self, problem: str, messages: List[Dict[str, str]], max_attempts: int = 5):
        reasoning_process = ReasoningProcess(initial_problem=problem)
        web_info = web_viewer(problem)

        chosen_prompt = self.prompt_optimizer.choose_prompt(problem, messages)

        # 获取用户的上一个问题（如果存在）
        previous_question = None
        for msg in reversed(messages[:-1]):  # 跳过最后一条消息，因为它是当前问题
            if msg["role"] == "user":
                previous_question = msg["content"]
                break

        yield "思考中...\n正在进行初步分析"
        reasoning_process.ia_analysis, initial_steps = self.initial_analysis(problem, web_info, messages, chosen_prompt, previous_question)
        yield "初步分析完成。\n开始尝试解决问题..."

        for attempt_number in range(1, max_attempts + 1):
            steps = [Step(content=step.content, description=step.description) for step in initial_steps]
            yield f"正在进行第 {attempt_number} 次尝试"

            for i, step in enumerate(steps):
                yield f"正在进行第 {attempt_number} 次尝试 (步骤 {i+1}/{len(steps)})"
                step.result = self.execute_step(problem, step, web_info, i+1, steps[:i], messages, previous_question)

            yield f"第 {attempt_number} 次尝试完成，正在总结结果"
            final_reasoning = self.summarize_results(problem, steps)

            yield f"正在判断第 {attempt_number} 次尝试的结果"
            judgment = self.judge_result(problem, reasoning_process.ia_analysis, steps, final_reasoning, web_info, chosen_prompt)

            attempt = Attempt(number=attempt_number, steps=steps, final_reasoning=final_reasoning, judgment=judgment)
            reasoning_process.attempts.append(attempt)

            if self.is_result_correct(judgment):
                yield "正在生成最终回答"
                break

        reasoning_process.final_answer = self.final_answer_fa(problem, reasoning_process.attempts[-1].final_reasoning, reasoning_process.attempts[-1].judgment)
        self.last_fa_output = reasoning_process.final_answer

        reward = self.calculate_reward(reasoning_process)
        self.prompt_optimizer.update(chosen_prompt, reward)

        yield reasoning_process

    def calculate_reward(self, reasoning_process):
        if self.is_result_correct(reasoning_process.attempts[-1].judgment):
            return 10 - len(reasoning_process.attempts)
        else:
            return max(-1, 5 - len(reasoning_process.attempts))

    def initial_analysis(self, problem: str, web_info: str, messages: List[Dict[str, str]], chosen_prompt: str, previous_question: Optional[str]) -> Tuple[str, List[Step]]:
        prompt = f"""
{chosen_prompt}

当前问题：
{problem}

上一个问题：
{previous_question if previous_question else "无"}

网络搜索信息：
{web_info}

注意：
1. 请提供你的分析和逐步计划。你的分析应该紧密围绕当前问题"{problem}"，并考虑上一个问题（如果有）。
2. 逐步计划应该详细到每个步骤，但尽量保持步骤数量最少。
3. 每个步骤应独立且不重复，避免不必要的重复处理。
4. 请不要给出最终答案，只需提供分析和计划。
5. 如果不需要网络搜索信息，请忽略。
6. 请严格按照以下XML格式返回结果，确保XML格式正确无误，否则会导致解析失败：

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

请确保你的回答只包含上述XML格式的内容，不要添加任何其他文本。
        """

        analysis_messages = messages + [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model=model,
            messages=analysis_messages,
            temperature=0.3
        ).choices[0].message.content.strip()

        steps = self.parse_steps_xml(response)
        return response, steps

    def parse_steps_xml(self, analysis: str) -> List[Step]:
        xml_content = re.search(r'<analysis>.*</analysis>', analysis, re.DOTALL)
        if xml_content:
            analysis = xml_content.group(0)

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
            old_steps = re.findall(r'^\d+\.\s*(.+)', analysis, re.MULTILINE)
            return [Step(content=step, description="") for step in old_steps]
        return [Step(content=content.strip(), description=description.strip()) for content, description in steps]

    def execute_step(self, problem: str, step: Step, web_info: str, step_number: int, previous_steps: List[Step], messages: List[Dict[str, str]], previous_question: Optional[str]) -> str:
        previous_steps_info = "\n".join([f"步骤 {i+1}：\n内容：{s.content}\n结果：{s.result}" for i, s in enumerate(previous_steps)])

        prompt = f"""
根据以下信息，执行任务流程并完成步骤 {step_number}：

当前问题：{problem}

上一个问题：{previous_question if previous_question else "无"}

步骤内容：{step.content}

步骤指导：{step.description}

网络搜索信息：{web_info}

之前的步骤：
{previous_steps_info}

请执行以下任务：
1. 回顾之前所有步骤的推理结果。
2. 考虑上一个问题（如果有）与当前问题的关系。
3. 质疑和验证之前步骤的正确性。如果发现任何错误或不一致，请指出并解释。
4. 如果需要，对之前的步骤进行修正。
5. 结合以上信息，详细完成当前步骤，并给出相应的结果。

请以下面的格式提供你的回答：

回顾与验证：
[在这里提供对之前步骤的回顾、质疑和验证]

修正（如果需要）：
[如果需要修正之前的步骤，请在这里说明]

当前步骤执行：
[在这里详细说明当前步骤的执行过程和结果]
        """
        step_messages = messages + [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model=model,
            messages=step_messages,
            temperature=0.7
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
1. 请确保推理过程合理，步骤清晰，结论有充分的依据支持。
2. 请仔细检查推理过程中是否有任何明显的错误或遗漏。
3. 请确保答案与原始问题直接相关。
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

    def judge_result(self, problem: str, ia_analysis: str, steps: List[Step], final_reasoning: str, web_info: str, chosen_prompt: str) -> str:
        combined_steps = "\n".join([f"步骤 {i+1} 描述：{step.description}\n步骤 {i+1} 结果：{step.result}" for i, step in enumerate(steps)])
        prompt = f"""
请综合以下信息，判断当前的推理结果是否正确，是否需要进一步迭代：

原始问题：{problem}

选择的初始 prompt：{chosen_prompt}

初步分析：{ia_analysis}

步骤及结果：
{combined_steps}

最终推理：
{final_reasoning}

网络搜索信息：
{web_info}

请给出你的判断，考虑以下几点：
1. 解决方案是否直接回答了原始问题？
2. 解决过程是否符合初始 prompt 的要求和方向？
3. 推理过程是否合理，步骤是否清晰？
4. 最终结论是否有充分的依据支持？
5. 是否有任何明显的错误或遗漏？

如果认为答案已经正确且完整，请明确说明"结果正确"，并简要总结正确的答案。
如果认为还需要进一步迭代，请详细说明原因和改进建议。
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

solver = ProblemSolver()

def user_input(user_message, history):
    return "", history + [[user_message, None]]

def chat(history):
    history_copy = history.copy()
    message = history_copy[-1][0]
    history_copy[-1][1] = "思考中..."
    yield history_copy

    messages = [
        {"role": "system", "content": get_system_prompt()}
    ]

    # 添加历史消息
    for user_msg, assistant_msg in history_copy[:-1]:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            messages.append({"role": "assistant", "content": assistant_msg})

    # 添加当前用户消息
    messages.append({"role": "user", "content": message})

    for update in solver.solve_problem_with_history(message, messages):
        if isinstance(update, str):
            history_copy[-1][1] = update
            yield history_copy
        elif isinstance(update, ReasoningProcess):
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
