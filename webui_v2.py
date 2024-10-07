import os
import gradio as gr
import re
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from datetime import datetime


# 假设这是您用于获取网络搜索信息的模块
from webviewer import web_viewer

def get_api_key(fp_or_env):
    if os.path.exists(fp_or_env):
        with open(fp_or_env, "r") as f:
            api_key = f.read().strip()
    else:
        api_key = os.getenv(fp_or_env)
    return api_key

# 初始化 OpenAI 客户端
glm_api_key = get_api_key("api_key") or get_api_key("API_KEY")
glm_client = OpenAI(
    api_key=glm_api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)
glm_model = "glm-4-plus"
glm_step_model = "glm-4-flash"
glm_step_client = OpenAI(
    api_key=glm_api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

def get_current_time():
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")

def get_system_prompt():
    current_time = get_current_time()
    return f"""你是一个名为"思考者"（Thinker）的AI助手，非常擅长解决复杂的STEM问题。请使用与用户提问语言相同的语言回答，或者直接使用用户指定的语言。
当前时间：{current_time}"""

@dataclass
class Attempt:
    attempt_number: int
    summary: str
    reasoning: str
    response: str
    score: int

@dataclass
class ReasoningProcess:
    initial_problem: str
    attempts: List[Attempt] = field(default_factory=list)
    final_answer: Optional[str] = None

class ProblemSolver:
    def __init__(self):
        self.last_fa_output = None
        self.reasoning_process = None

    # 更新计算器函数，支持科学计算
    def calculator(self, tool_input: str) -> str:
        try:
            import ast
            import operator as op
            import math

            allowed_operators = {
                ast.Add: op.add,
                ast.Sub: op.sub,
                ast.Mult: op.mul,
                ast.Div: op.truediv,
                ast.Pow: op.pow,
                ast.Mod: op.mod,
                ast.UAdd: op.pos,
                ast.USub: op.neg,
            }

            allowed_functions = {
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'sqrt': math.sqrt,
                'log': math.log,
                'log10': math.log10,
                'exp': math.exp,
                'pow': math.pow,
                'fabs': math.fabs,
                'factorial': math.factorial,
                'floor': math.floor,
                'ceil': math.ceil,
                'degrees': math.degrees,
                'radians': math.radians,
            }

            def eval_expr(node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    return allowed_operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return allowed_operators[type(node.op)](eval_expr(node.operand))
                elif isinstance(node, ast.Call):
                    func_name = node.func.id
                    if func_name in allowed_functions:
                        args = [eval_expr(arg) for arg in node.args]
                        return allowed_functions[func_name](*args)
                    else:
                        raise TypeError(f"不支持的函数: {func_name}")
                else:
                    raise TypeError(f"不支持的类型: {type(node)}")

            expr = ast.parse(tool_input, mode='eval').body
            result = eval_expr(expr)
            return str(result)
        except Exception as e:
            return f"计算错误：{e}"

    # 实现搜索引擎工具
    def search_engine(self, tool_input: str) -> str:
        result = web_viewer(tool_input)
        return result

    # 实现代码解释器工具
    def code_interpreter(self, tool_input: str) -> str:
        try:
            exec_globals = {}
            exec_locals = {}
            exec(tool_input, exec_globals, exec_locals)
            output = exec_locals.get('output', '代码已执行，但未找到变量 output。')
            return str(output)
        except Exception as e:
            return f"代码执行错误：{e}"

    def execute_tool(self, tool_name: str, tool_input: str) -> str:
        if tool_name == "计算器":
            return self.calculator(tool_input)
        elif tool_name == "搜索引擎":
            return self.search_engine(tool_input)
        elif tool_name == "代码解释器":
            return self.code_interpreter(tool_input)
        else:
            return f"未知的工具：{tool_name}"

    def solve_problem_with_history(self, problem: str, messages: List[Dict[str, str]], max_attempts: int = 5):
        self.reasoning_process = ReasoningProcess(initial_problem=problem)
        web_info = self.search_engine(problem)  # 使用搜索引擎工具获取网络搜索信息

        attempt_number = 1
        while attempt_number <= max_attempts:
            # Planner 规划
            summary, reasoning = self.planner_reasoning(problem, web_info, messages)
            if not summary or not reasoning:
                yield "未能提供有效的推理方向，结束。"
                break
            messages.append({"role": "assistant", "content": f"<reason_summary>{summary}</reason_summary><reason>{reasoning}</reason>"})

            # 直接输出规划者的摘要
            yield f"思考步骤摘要：{summary}"

            # 保存详细规划到尝试过程
            attempt = Attempt(
                attempt_number=attempt_number,
                summary=summary,
                reasoning=reasoning,
                response="",
                score=0
            )
            self.reasoning_process.attempts.append(attempt)

            # Executor 执行
            response, score = self.executor_execute(problem, reasoning, web_info, messages)
            if score is None:
                yield "未能提供有效的评分，结束。"
                break
            messages.append({"role": "assistant", "content": response})

            # 更新尝试过程
            attempt.response = response
            attempt.score = score

            # 判断是否需要进一步尝试
            if score >= 4:
                # Planner 总结并给出最终答案
                final_answer = self.planner_finalize(problem, reasoning, response, web_info, messages)
                self.reasoning_process.final_answer = final_answer
                yield self.reasoning_process  # 返回完整的推理过程
                break
            else:
                # 引导规划者尝试不同的策略或分解问题
                messages.append({"role": "user", "content": "请尝试另一种方法，或者将问题分解后再尝试解决。"})
                attempt_number += 1
                continue  # 继续下一次尝试
        else:
            yield "已达到最大尝试次数，结束。"
            # 如果在最大尝试次数后仍未解决问题，可以给出总结或提示
            final_answer = self.planner_finalize(problem, reasoning, response, web_info, messages)
            self.reasoning_process.final_answer = final_answer
            yield self.reasoning_process  # 返回完整的推理过程

    def planner_reasoning(self, problem: str, web_info: str, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        previous_attempts = "\n".join(
            [f"尝试 {a.attempt_number} 的思考步骤：{a.reasoning}，评分：{a.score}" for a in self.reasoning_process.attempts]
        )
        prompt = f"""
请根据以下问题、相关的网络搜索信息和之前的尝试，规划你将要采取的新的解决问题的思路。请避免重复之前的错误，并尝试新的方法。请先给出思考步骤的摘要，然后提供详细的规划。请使用指定的 XML 标签返回结果：

问题：{problem}

网络搜索信息：{web_info}

之前的尝试：
{previous_attempts}

当前可用的工具和主要用途：
**工具列表：**

1. **计算器**：用于执行基本和科学计算，包括加减乘除、幂、对数、三角函数等常见数学运算。适用于**单个数学表达式**的计算。

2. **搜索引擎**：用于获取最新的信息、定义和一般知识。适用于**查询事实**和**获取信息**。

3. **代码解释器**：用于编写和执行代码，适用于无法通过计算器直接计算的**复杂计算**、**数据处理**、**模拟**等。

**工具选择指南：**

- **当需要计算单个数学表达式时，使用计算器。**

- **当需要编写代码解决复杂问题或进行数据处理时，使用代码解释器。**

- **当需要查询知识、定义或事实信息时，使用搜索引擎。**


请严格按照以下格式返回你的规划：
<reason_summary>思考步骤摘要</reason_summary>
<reason>你的思考步骤</reason>

注意：
- 不要直接回答问题本身。
- 请描述你将如何解决问题，先提供摘要，再给出详细的步骤。如果可以使用工具，请规划使用工具的步骤。
- 使用第一人称，说明你将要做什么。
- 不要包含判断、反思或总结内容。
- 请不要添加任何额外的文本或说明。
        """
        response = glm_client.chat.completions.create(
            model=glm_model,
            messages=messages + [{"role": "user", "content": prompt}],
            temperature=0.7
        ).choices[0].message.content.strip()

        # 提取 <reason_summary> 和 <reason> 标签中的内容
        summary_match = re.search(r'<reason_summary>(.*?)</reason_summary>', response, re.DOTALL)
        reason_match = re.search(r'<reason>(.*?)</reason>', response, re.DOTALL)

        if summary_match and reason_match:
            summary = summary_match.group(1).strip()
            reason = reason_match.group(1).strip()
            return summary, reason
        else:
            # 如果未找到标签，处理异常
            return None, None

    def executor_execute(self, problem: str, reasoning: str, web_info: str, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        prompt = f"""
请根据以下推理方向，尝试解决问题，并详细描述你的思考过程。

**工具列表：**

1. **计算器**：用于执行基本和科学计算，包括加减乘除、幂、对数、三角函数等常见数学运算。适用于**单个数学表达式**的计算。

2. **搜索引擎**：用于获取最新的信息、定义和一般知识。适用于**查询事实**和**获取信息**。

3. **代码解释器**：用于编写和执行代码，适用于无法通过计算器直接计算的**复杂计算**、**数据处理**、**模拟**等。

**工具选择指南：**

- **当需要计算单个数学表达式时，使用计算器。**

- **当需要编写代码解决复杂问题或进行数据处理时，使用代码解释器。**

- **当需要查询知识、定义或事实信息时，使用搜索引擎。**

**使用工具的格式：**

当在思考过程中需要使用工具，请使用以下格式请求：

<tool>
工具名称: 工具名称
输入: 工具的输入内容
</tool>

我们会执行工具并将结果返回给你。

**开始任务：**

问题：{problem}

推理方向：{reasoning}

网络搜索信息：{web_info}

**要求：**

1. 逐步展开你的推理，每一步都清晰呈现。

2. **根据工具选择指南，选择最适合的工具，并使用工具请求格式来请求工具的帮助。**

3. 在每个步骤后，检查并纠正可能的错误。

4. 如果发现前面的思路有误，尝试新的策略。

5. 最终得出你的答案。

6. 在答案末尾，用 XML 格式返回你对解决方案的评分（1-5 分），使用标签 <score>score</score>。

格式示例：
<score>5</score>

**注意：**

- 评分标准：答案可信度越高，得分越高。

- 请认真反思你的答案，并根据自信程度进行评分。

- **请不要在评分标签之外添加评分的解释或描述。**

- **请确保评分标签单独占一行。**
    """
        response = glm_step_client.chat.completions.create(
            model=glm_step_model,
            messages=messages + [{"role": "user", "content": prompt}],
            temperature=0.5  # 调低温度以提高指令遵循性
        ).choices[0].message.content.strip()

        # 检查并处理工具请求
        while True:
            tool_request_match = re.search(r'<tool>(.*?)</tool>', response, re.DOTALL)
            if tool_request_match:
                tool_request = tool_request_match.group(1).strip()
                # 解析工具请求
                tool_name_match = re.search(r'工具名称:\s*(.*)', tool_request)
                tool_input_match = re.search(r'输入:\s*(.*)', tool_request)
                if tool_name_match and tool_input_match:
                    tool_name = tool_name_match.group(1).strip()
                    tool_input = tool_input_match.group(1).strip()
                    # 执行工具
                    tool_result = self.execute_tool(tool_name, tool_input)
                    # 将工具结果返回给模型
                    tool_result_message = f"工具 {tool_name} 的结果：{tool_result}"
                    messages.append({"role": "assistant", "content": tool_result_message})
                    # 更新提示，提醒模型包含评分
                    prompt = f"""
请根据工具的结果继续你的推理，并记得移除已使用的工具请求。

请记住以下要求：
1. 继续你的推理，每一步都清晰呈现。

2. 在适当的时候，检查并纠正可能的错误。

3. 最终得出你的答案。

4. 在答案末尾，用 XML 格式返回你对解决方案的评分（1-5 分），使用标签 <score>score</score>。

格式示例：
<score>5</score>

**注意：**

- 评分标准：答案可信度越高，得分越高。

- 请认真反思你的答案，并根据自信程度进行评分。

- **请不要在评分标签之外添加评分的解释或描述。**

- **请确保评分标签单独占一行。**
                """
                # 继续与模型对话
                response = glm_step_client.chat.completions.create(
                    model=glm_step_model,
                    messages=messages + [{"role": "user", "content": prompt}],
                    temperature=0.5
                ).choices[0].message.content.strip()
                continue  # 继续处理下一个工具请求
            else:
                break

        # 提取 <score> 标签中的评分
        score_match = re.search(r'<score>\s*(\d+)\s*</score>', response, re.IGNORECASE | re.DOTALL)
        if score_match:
            score = int(score_match.group(1))
            # 移除 <score> 标签，保留答案内容
            response = re.sub(r'<score>\s*(\d+)\s*</score>', '', response, flags=re.IGNORECASE | re.DOTALL).strip()
        else:
            score = None
        return response, score

    def planner_finalize(self, problem: str, reasoning: str, executor_response: str, web_info: str, messages: List[Dict[str, str]]) -> str:
        prompt = f"""
请根据以下内容，给出问题的最终答案：

问题：{problem}

你的回答：{executor_response}

注意：
- 请确保最终答案直接回答了原始问题。
- 可以补充必要的解释，但请尽量简洁明了。
- 不要包含判断、反思或总结内容。
- 如果需要，可以参考以下的网络搜索信息，但不要直接复制。

网络搜索信息：{web_info}
        """
        response = glm_step_client.chat.completions.create(
            model=glm_step_model,
            messages=messages + [{"role": "user", "content": prompt}],
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
            # 更新历史记录，显示规划者的摘要
            history_copy[-1][1] = update
            yield history_copy
        elif isinstance(update, ReasoningProcess):
            # 生成折叠框内容，展示完整的推理过程
            attempts_content = ""
            for attempt in update.attempts:
                attempts_content += f"""
<details>
    <summary>尝试 {attempt.attempt_number}</summary>
    <p><strong>思考步骤摘要：</strong>{attempt.summary}</p>
    <p><strong>详细思考步骤：</strong>{attempt.reasoning}</p>
    <p><strong>回答：</strong>{attempt.response}</p>
    <p><strong>评分：</strong>{attempt.score} 分</p>
</details>
                """
            collapsible_content = f"""
<details>
    <summary>详细规划和推理过程</summary>
    {attempts_content}
</details>
            """
            # 确保最终答案在折叠框之外
            final_answer = update.final_answer
            full_response = collapsible_content + "\n\n" + final_answer
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
