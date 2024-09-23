import os
import gradio as gr
import re
import xml.etree.ElementTree as ET
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from webviewer import web_viewer
from datetime import datetime
from bs4 import BeautifulSoup


def get_api_key(fp_or_env):
    if os.path.exists(fp_or_env):
        with open(fp_or_env, "r") as f:
            api_key = f.read().strip()
    else:
        api_key = os.getenv(fp_or_env)
    return api_key


# 初始化 OpenAI 客户端
glm_api_key = get_api_key("api_key") or get_api_key("API_KEY")
deepseek_api_key = get_api_key("deepseek_api_key") or get_api_key("DEEPSEEK_API_KEY")

glm_client = OpenAI(
    api_key=glm_api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)
glm_model = "glm-4-plus"
glm_step_model = "glm-4-flash"

deepseek_client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com/v1"
)
deepseek_model = "deepseek-chat"



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
    prompt_selection: str = ""
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
            PromptTemplate("作为一名经验丰富的教师，你会如何设计一个课程来帮助学生理解和解决这个{problem_type}问题？", ["teacher"]),
            PromptTemplate("你是一位政策制定者，面对这个{problem_type}问题，你会提出什么样的政策建议？请考虑各方面的影响。", ["policy-maker"]),
            PromptTemplate("作为一名企业家，你如何将这个{problem_type}问题转化为一个商业机会？请提出一个创新的商业模式。", ["entrepreneur"]),
            PromptTemplate("你是一位社会工作者，遇到了这个{problem_type}问题。你会如何帮助受影响的人群？请提供具体的援助计划。", ["social-worker"]),
            PromptTemplate("作为一名记者，你将如何报道这个{problem_type}问题？请提供一个全面、客观的报道框架。", ["journalist"]),
            PromptTemplate("你是一位环保主义者，面对这个{problem_type}问题，你会提出什么样的环保解决方案？请考虑可持续发展。", ["environmentalist"]),
            PromptTemplate("作为一名心理咨询师，你如何帮助受到这个{problem_type}问题影响的人们？请提供一些心理辅导策略。", ["psychologist"]),
            PromptTemplate("你是一位法律专家，这个{problem_type}问题可能涉及哪些法律问题？请提供相关的法律分析和建议。", ["legal-expert"]),
            PromptTemplate("作为一名艺术家，你会如何通过艺术创作来表达或解决这个{problem_type}问题？请描述你的创作理念。", ["artist"]),
            PromptTemplate("你是一位医疗专业人士，这个{problem_type}问题可能对人们的健康产生什么影响？请提供医学角度的分析和建议。", ["healthcare-professional"]),
            PromptTemplate("作为一名算法工程师,你会如何设计一个高效的算法来解决这个{problem_type}问题?请详细说明算法的思路、时间复杂度和空间复杂度。", ["algorithm-engineer"]),
            PromptTemplate("你是一位前端开发工程师,面对这个{problem_type}问题,你会如何设计用户界面和交互?请考虑用户体验和性能优化。", ["frontend-developer"]),
            PromptTemplate("作为一名后端开发工程师,你如何设计服务器端架构来处理这个{problem_type}问题?请考虑系统的可扩展性和安全性。", ["backend-developer"]),
            PromptTemplate("你是一位系统架构师,面对这个{problem_type}问题,你会提出什么样的整体架构设计?请考虑系统的可靠性、性能和可维护性。", ["system-architect"]),
            PromptTemplate("作为一名数据库专家,你会如何设计数据模型和优化查询来解决这个{problem_type}问题?请考虑数据一致性和查询效率。", ["database-expert"]),
            PromptTemplate("你是一位网络安全专家,这个{problem_type}问题可能涉及哪些安全风险?请提供相应的安全防护措施。", ["cybersecurity-expert"]),
            PromptTemplate("作为一名人工智能研究员,你会如何应用机器学习或深度学习技术来解决这个{problem_type}问题?请描述可能的模型和训练策略。", ["ai-researcher"]),
            PromptTemplate("你是一位DevOps工程师,面对这个{problem_type}问题,你会如何设计持续集成和部署流程?请考虑自动化和监控方面的需求。", ["devops-engineer"]),
            PromptTemplate("作为一名云计算专家,你会如何利用云服务来解决这个{problem_type}问题?请考虑成本效益和资源管理。", ["cloud-computing-expert"]),
            PromptTemplate("你是一位移动应用开发者,如何在移动平台上实现这个{problem_type}问题的解决方案?请考虑不同操作系统的兼容性。", ["mobile-app-developer"]),
        ]

    def choose_prompt(self, problem, messages):
        analysis = self.analyze_problem(problem)

        formatted_templates = []
        for template in self.prompt_templates:
            try:
                formatted_template = template.template.format(**analysis)
                formatted_templates.append(formatted_template)
            except KeyError as e:
                print(f"警告：模板格式化失败，缺少键 {e}。使用原始模板。")
                formatted_templates.append(template.template)

        prompt = f"""
给定以下问题：

{problem}

以及以下可能的prompt模板：

{formatted_templates}

请选择最适合解决这个问题的prompt模板。你的回答应该包含所选模板的编号（从1开始）和简短解释。例如：
"3 - 这个模板最适合，因为它强调了教育性解释，适合{analysis.get('level', '未知')}级别的{analysis.get('problem_type', '未知类型')}问题。"
        """

        response = glm_client.chat.completions.create(
            model=glm_model,
            messages=messages + [{"role": "user", "content": prompt}],
            temperature=0.3
        ).choices[0].message.content.strip()

        try:
            chosen_index = int(response.split()[0]) - 1
            chosen_template = self.prompt_templates[chosen_index]
        except (ValueError, IndexError):
            # 如果大模型的回答无效，就使用默认的选择方法
            chosen_template = max(self.prompt_templates, key=lambda t: t.performance_score)

        chosen_template.use_count += 1
        return chosen_template.template.format(**analysis), response

    def update(self, prompt, reward):
        for template in self.prompt_templates:
            if template.template in prompt:
                template.performance_score = (template.performance_score * template.use_count + reward) / (template.use_count + 1)
                break

    def analyze_problem(self, problem):
        prompt = f"""
请分析以下问题，并提供问题类型、所属领域和难度级别：

问题：{problem}

请以以下格式回答：
问题类型：[填写]
所属领域：[填写]
难度级别：[填写]

注意：
- 问题类型应该是一个简短的描述，如"数学"、"物理"、"编程"等。
- 所属领域应该更具体，如"代数"、"力学"、"算法"等。
- 难度级别应该是"小学"、"初中"、"高中"、"大学"或"研究生"中的一个。
        """

        response = glm_client.chat.completions.create(
            model=glm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        ).choices[0].message.content.strip()

        analysis = {
            'problem_type': '一般',
            'field': '综合',
            'level': '成人'
        }

        for line in response.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                if key == '问题类型':
                    analysis['problem_type'] = value
                elif key == '所属领域':
                    analysis['field'] = value
                elif key == '难度级别':
                    analysis['level'] = value

        return analysis

class ProblemSolver:
    def __init__(self):
        self.last_fa_output = None
        self.prompt_optimizer = PromptOptimizer()
        self.reasoning_process = None  # 初始化为 None

    def solve_problem_with_history(self, problem: str, messages: List[Dict[str, str]], max_attempts: int = 5):
        self.reasoning_process = ReasoningProcess(initial_problem=problem)  # 在这里初始化 reasoning_process
        web_info = web_viewer(problem)

        chosen_prompt, prompt_explanation = self.prompt_optimizer.choose_prompt(problem, messages)
        self.reasoning_process.prompt_selection = prompt_explanation

        # 获取用户的上一个问题（如果存在）
        previous_question = None
        for msg in reversed(messages[:-1]):  # 跳过最后一条消息，因为它是当前问题
            if msg["role"] == "user":
                previous_question = msg["content"]
                break

        yield "思考中...\n正在进行初步分析"
        self.reasoning_process.ia_analysis, initial_steps = self.initial_analysis(problem, web_info, messages, chosen_prompt, previous_question)
        yield "初步分析完成。\n开始尝试解决问题..."

        # 判断是否为计算机领域的问题
        is_computer_related = self.is_computer_related(problem)
        client = deepseek_client if is_computer_related else glm_client
        model = deepseek_model if is_computer_related else glm_model
        step_model = deepseek_model if is_computer_related else glm_step_model

        for attempt_number in range(1, max_attempts + 1):
            steps = [Step(content=step.content, description=step.description) for step in initial_steps]
            yield f"正在进行第 {attempt_number} 次尝试"

            for i, step in enumerate(steps):
                yield f"正在进行第 {attempt_number} 次尝试 (步骤 {i+1}/{len(steps)})"
                step.result = self.execute_step(problem, step, web_info, i+1, steps[:i], messages, previous_question, client, step_model)

            yield f"第 {attempt_number} 次尝试完成，正在总结结果"
            final_reasoning = self.summarize_results(problem, steps, client, model)

            yield f"正在判断第 {attempt_number} 次尝试的结果"
            judgment = self.judge_result(problem, self.reasoning_process.ia_analysis, steps, final_reasoning, web_info, chosen_prompt, client, model)

            attempt = Attempt(number=attempt_number, steps=steps, final_reasoning=final_reasoning, judgment=judgment)
            self.reasoning_process.attempts.append(attempt)

            if self.is_result_correct(judgment):
                yield "正在生成最终回答"
                break

        self.reasoning_process.final_answer = self.final_answer_fa(problem, self.reasoning_process.attempts[-1].final_reasoning, self.reasoning_process.attempts[-1].judgment, client, model)
        self.last_fa_output = self.reasoning_process.final_answer

        reward = self.calculate_reward(self.reasoning_process)
        self.prompt_optimizer.update(chosen_prompt, reward)

        yield self.reasoning_process

    def is_computer_related(self, problem: str) -> bool:
        prompt = f"""
请判断以下问题是否属于计算机领域：

问题：{problem}

如果问题属于计算机领域，请回答"是"；否则回答"否"。
        """
        response = glm_client.chat.completions.create(
            model=glm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        ).choices[0].message.content.strip().lower()

        return "是" in response

    def execute_step(self, problem: str, step: Step, web_info: str, step_number: int, previous_steps: List[Step], messages: List[Dict[str, str]], previous_question: Optional[str], client, model) -> str:
        previous_steps_info = "\n".join([f"步骤 {i+1}：\n内容：{s.content}\n结果：{s.result}" for i, s in enumerate(previous_steps)])

        previous_attempts_info = ""
        for i, attempt in enumerate(self.reasoning_process.attempts):
            previous_attempts_info += f"\n尝试 {i+1}：\n"
            previous_attempts_info += "\n".join([f"步骤 {j+1}：\n内容：{s.content}\n结果：{s.result}" for j, s in enumerate(attempt.steps)])
            previous_attempts_info += f"\n最终推理：{attempt.final_reasoning}\n判断：{attempt.judgment}\n"

        prompt = f"""
根据以下信息，执行任务流程并完成步骤 {step_number}：

当前问题：{problem}

上一个问题：{previous_question if previous_question else "无"}

步骤内容：{step.content}

步骤指导：{step.description}

网络搜索信息：{web_info}

之前的步骤：
{previous_steps_info}

之前的尝试：
{previous_attempts_info}

请执行以下任务：
1. 回顾之前所有步骤和尝试的推理结果。
2. 考虑上一个问题（如果有）与当前问题的关系。
3. 分析之前的尝试为什么失败，指出关键问题和错误。
4. 基于之前尝试的失败经验，提出改进建议。
5. 质疑和验证之前步骤的正确性。如果发现任何错误或不一致，请指出并解释。
6. 如果需要，对之前的步骤进行修正。
7. 结合以上信息，详细完成当前步骤，并给出相应的结果。

请以下面的格式提供你的回答：

回顾与验证：
[在这里提供对之前步骤和尝试的回顾、质疑和验证]

失败分析：
[分析之前尝试失败的原因]

改进建议：
[基于失败分析提出的改进建议]

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

    def summarize_results(self, problem: str, steps: List[Step], client, model) -> str:
        combined_results = "\n".join([f"步骤 {i+1} 结果：{step.result}" for i, step in enumerate(steps)])
        prompt = f"""
请综合以下步骤的结果，进行最终的推理：

问题：{problem}

步骤结果：
{combined_results}

请根据这些信息，给出一个综合的推理结论。

注意：
1. 请确保推理过程合理，步骤清晰，结论有充分的依据支持。
2. 请遵循步骤处理结果，仔细检查推理过程中是否有任何明显的错误或遗漏。
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

    def judge_result(self, problem: str, ia_analysis: str, steps: List[Step], final_reasoning: str, web_info: str, chosen_prompt: str, client, model) -> str:
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

    def final_answer_fa(self, problem: str, final_reasoning: str, judgment: str, client, model) -> str:
        prompt = f"""
请根据以下信息，整合出一个简洁的最终答案：

原始问题：{problem}

最终推理：{final_reasoning}

判断结果：{judgment}

请给出一个简洁的最终答案，确保答案与原始问题直接相关。如果你认为需要向用户解释，请在答案中添加解释。
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
6. 请严格按照以下XML格式返回结果，确保XML格式正确无误：

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

        response = glm_client.chat.completions.create(
            model=glm_model,
            messages=analysis_messages,
            temperature=0.3
        ).choices[0].message.content.strip()

        # 添加后处理步骤
        response = self.fix_xml(response)

        steps = self.parse_steps_xml(response)
        return response, steps

    def fix_xml(self, xml_string: str) -> str:
        # 移除所有非XML内容
        xml_content = re.search(r'<analysis>.*</analysis>', xml_string, re.DOTALL)
        if xml_content:
            xml_string = xml_content.group(0)
        else:
            # 如果没有找到完整的<analysis>标签，尝试提取所有<step>标签
            steps = re.findall(r'<step>.*?</step>', xml_string, re.DOTALL)
            if steps:
                xml_string = f"<analysis>{''.join(steps)}</analysis>"
            else:
                # 如果还是找不到，返回一个空的analysis标签
                return "<analysis></analysis>"

        # 移除XML声明（如果存在）
        xml_string = re.sub(r'<\?xml.*?\?>', '', xml_string)

        # 确保所有标签都正确关闭
        xml_string = re.sub(r'<(\w+)([^>]*)>', r'<\1\2></\1>', xml_string)
        xml_string = re.sub(r'</(\w+)>(?=.*</\1>)', '', xml_string)

        # 移除重复的结束标签
        xml_string = re.sub(r'(</\w+>)\1+', r'\1', xml_string)

        # 确保只有一个根元素
        xml_string = re.sub(r'</analysis>.*?<analysis>', '', xml_string)

        # 如果没有根元素，添加一个
        if not xml_string.startswith('<analysis>'):
            xml_string = f'<analysis>{xml_string}</analysis>'

        return xml_string.strip()

    def parse_steps_xml(self, analysis: str) -> List[Step]:
        # 步骤1: 尝试修复常见的XML错误
        analysis = self.fix_xml(analysis)

        # 步骤2: 尝试使用 xml.etree.ElementTree 解析
        try:
            root = ET.fromstring(analysis)
            steps = []
            for step_elem in root.findall('.//step'):
                content = step_elem.find('content')
                description = step_elem.find('description')
                content_text = content.text.strip() if content is not None and content.text else ""
                description_text = description.text.strip() if description is not None and description.text else ""
                steps.append(Step(content=content_text, description=description_text))
            if steps:
                return steps
        except ET.ParseError as e:
            pass

        # 步骤3: 尝试使用 BeautifulSoup 解析
        try:
            soup = BeautifulSoup(analysis, 'xml')
            steps = []
            for step_elem in soup.find_all('step'):
                content = step_elem.find('content')
                description = step_elem.find('description')
                content_text = content.text.strip() if content else ""
                description_text = description.text.strip() if description else ""
                steps.append(Step(content=content_text, description=description_text))
            if steps:
                return steps
        except Exception as e:
            print("BeautifulSoup解析错误:", e)

        # 步骤4: 如果所有方法都失败，使用正则表达式
        return self.parse_steps_fallback(analysis)

    def parse_steps_fallback(self, analysis: str) -> List[Step]:
        steps = []
        step_pattern = r'<step>\s*<content>(.*?)</content>\s*<description>(.*?)</description>\s*</step>'
        matches = re.findall(step_pattern, analysis, re.DOTALL)

        if not matches:
            # 如果找不到完整的step结构，尝试单独匹配content和description
            content_matches = re.findall(r'<content>(.*?)</content>', analysis, re.DOTALL)
            description_matches = re.findall(r'<description>(.*?)</description>', analysis, re.DOTALL)

            # 使用找到的内容创建Step对象
            for i in range(max(len(content_matches), len(description_matches))):
                content = content_matches[i].strip() if i < len(content_matches) else ""
                description = description_matches[i].strip() if i < len(description_matches) else ""
                steps.append(Step(content=content, description=description))
        else:
            for content, description in matches:
                steps.append(Step(content=content.strip(), description=description.strip()))

        # 如果仍然没有找到任何步骤，尝试使用旧的方法
        if not steps:
            old_steps = re.findall(r'^\d+\.\s*(.+)', analysis, re.MULTILINE)
            steps = [Step(content=step.strip(), description="") for step in old_steps]

        return steps

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
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            show_label=False,
            height=500
        )

        with gr.Row():
            msg = gr.Textbox(show_label=False, placeholder="输入您的问题...", scale=8)
            send_button = gr.Button("发送", scale=1)

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
