# ref: https://mp.weixin.qq.com/s/6OGDLnn0VVXOAHV4G2SQww

from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Optional
from IPython.display import display, Markdown, HTML
from webviewer import web_viewer
from datetime import datetime


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
    return f"""你是一个名为"思考者"（Thinker）的AI助手，正在继续解决问题的过程。请使用与用户提问语言相同的语言回答，或者直接使用用户指定的语言。
当前时间：{current_time}"""

@dataclass
class JudgmentResult:
    is_correct: bool
    score: float
    feedback: str
    use_web_info: bool
    accuracy_summary: str
    deficiency_summary: str
    self_reflection: str  # 新增字段

@dataclass
class ThoughtStep:
    step_answer: str
    is_completed: bool
    hint: str
    judgment: Optional[JudgmentResult] = None

@dataclass
class ReasoningProcess:
    initial_problem: str
    steps: List[ThoughtStep] = field(default_factory=list)
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

def solve_problem(problem: str, max_attempts: int = 10) -> ReasoningProcess:
    bilingual_display("Problem", "问题")
    display(Markdown(f"{problem}\n"))
    
    reasoning_process = ReasoningProcess(initial_problem=problem)
    attempts = 0
    is_completed = False
    min_attempts = 3  # 最少尝试次数

    # 进行一次网络搜索，结果将在整个过程中复用
    web_info = web_viewer(problem)

    # 步骤1：分问题并制定计划
    analysis_prompt = f"""
作为一个擅长解决复杂STEM问题的思考者（Thinker），请使用多步骤推理来解决以下问题。
首先分析问题，思考可能的解决方法，并规划后续解决步骤。
请参考提供的网络搜索信息。

问题：
{problem}

网络搜索信息：
{web_info}

请用简洁的文字提供你的分析和逐步计划。
"""

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": analysis_prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages
    ).choices[0].message.content.strip()

    # 显示AI的初步分析
    display_collapsible("Initial Answer Attempt and Step Breakdown / 初步答案尝试和步骤分解", response)

    hint = response
    analysis_step = ThoughtStep(step_answer="", is_completed=False, hint=hint)
    reasoning_process.steps.append(analysis_step)

    messages = [{"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": f"对这个问题进行思考，并参考以下网络搜索信息：\n\n问题：{problem}\n\n网络搜索信息：{web_info}"},
                {"role": "assistant", "content": hint},
                {"role": "user", "content": "根据这个思路和网络搜索信息解决问题，并给出答案。"}]

    # 继续执行计划并尝试解决问题
    while (not is_completed and attempts < max_attempts) or attempts < min_attempts:
        attempts += 1
        
        try:
            # 修改：在生成步骤答案的 prompt 中加入反问元素
            step_prompt = f"""
根据这个思路和网络搜索信息解决问题，并给出答案。

在给出答案后，请仔细思考并回答以下问题：
1. 这个答案是否完全正确？
2. 如果不完全正确，哪里需要改进？

请首先给出你的答案，然后回答上述问题。
"""
            messages.append({"role": "user", "content": step_prompt})

            response = client.chat.completions.create(
                model=model,
                messages=messages
            ).choices[0].message.content.strip()

            # 提取步骤答案和自我评估
            parts = response.split("\n\n", 1)
            step_answer = parts[0].strip()
            self_assessment = parts[1].strip() if len(parts) > 1 else ""

            display_collapsible(f"Step Answer (Attempt {attempts}) / 步骤答案（尝试 {attempts}）", step_answer)
            if self_assessment:
                display_collapsible(f"Self-Assessment (Attempt {attempts}) / 自我评估（尝试 {attempts}）", self_assessment)

            # 记录步骤答案
            step = ThoughtStep(step_answer=step_answer, is_completed=False, hint=hint)
            reasoning_process.steps.append(step)

            messages.append({"role": "assistant", "content": step_answer})

            # 如果达到最小尝试次数，进行评判
            if attempts >= min_attempts:
                # 使用评判者模型
                previous_steps = [step.step_answer for step in reasoning_process.steps]
                judgment = judge_step(problem, step_answer, previous_steps, web_info)
                step.judgment = judgment
                step.is_completed = judgment.is_correct

                judgment_content = f"""
是否正确：{'是' if judgment.is_correct else '否'}
得分：{judgment.score}/100
反馈：{judgment.feedback}
准确性总结：{judgment.accuracy_summary}
缺陷总结：{judgment.deficiency_summary}
自我反思：{judgment.self_reflection}
"""
                display_collapsible(f"Judgment Result (Attempt {attempts}) / 评判结果（尝试 {attempts}）", judgment_content)

                if judgment.is_correct and judgment.score >= 90:
                    is_completed = True
                    reasoning_process.final_answer = step_answer
                    break

                # 如果评判结果不理想，使用反馈作为新的提示
                if not is_completed:
                    hint = judgment.feedback
                    messages.append({"role": "user", "content": f"根据以下反馈改进你的答案，并继续思考问题的解决方案：{hint}"})
            else:
                # 如果还未达到最小尝试次数，继续思考
                messages.append({"role": "user", "content": "请继续思考并改进你的答案。"})

        except Exception as e:
            print(f"错误：在尝试 {attempts} 中发生异常：{str(e)}")
            continue

    # 如果达到最大尝试次数或循环结束后仍未得到答案，再次尝试获取最终答案
    if not reasoning_process.final_answer:
        bilingual_display("Max attempts reached or no satisfactory answer found", "达到最大尝试次数或未找到满意答案")
        
        # 准备所有步骤的总结
        all_steps_summary = "\n".join([f"步骤 {i+1}:\n{step.step_answer}\n评判:\n准确性：{step.judgment.accuracy_summary if step.judgment else '无评判'}\n缺陷：{step.judgment.deficiency_summary if step.judgment else '无评判'}" for i, step in enumerate(reasoning_process.steps)])
        
        final_prompt = f"""
基于之前的推理过程总结，请给出你的最终答案。

问题：
{reasoning_process.initial_problem}

推理过程总结：
{all_steps_summary}

请注意：
1. 使用与问题相同的语言回答。
2. 答案尽量简短，避免出现任何与问题"{reasoning_process.initial_problem}"不相关的答案。
3. 直接说出改进后的答案，而不需要解释，或者强调是改进后的答案。

请现在给出答案：
"""

        messages.append({"role": "user", "content": final_prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages
        ).choices[0].message.content.strip()

        reasoning_process.final_answer = response

    # 显示最终答案
    bilingual_display("Final Answer", "最终答案")
    final_answer = reasoning_process.final_answer
    display(Markdown(f"{final_answer}"))

    return reasoning_process

def judge_step(problem: str, step_answer: str, previous_steps: List[str], web_info: str) -> JudgmentResult:
    judgment_prompt = f"""
作为一个专业的评判者（思考者/Thinker），请评估以下问题解决步骤的正确性和质量。请务必参考提供的互联网信息。

问题：
{problem}

当前步骤答案：
{step_answer}

之前的步骤：
{chr(10).join(previous_steps)}

互联网信息：
{web_info}

请根据以下标准进行评判：
1. 逻辑一致性（0-20分）：思维步骤是否符合逻辑，没有矛盾。
2. 相关性（0-20分）：每一步是否与问题相关，并推进了解决方案。
3. 准确性（0-20分）：计算、事实或推理是否准确，是否与互联网信息一致。
4. 完整性（0-20分）：是否涵盖了问题的所有方面。
5. 创新性（0-10分）：解决方案是否有创新之处。
6. 可行性（0-10分）：解决方案是否实际可行。

请仔细评估答案，并提供详细的反馈。如果答案完全正确，请给出高分并说明原因。

在给出评判结果后，请反问自己：
1. 这个评判是否公正合理？
2. 是否有遗漏或误解的地方？

请按以下格式回答：

是否正确：[是/否]
得分：[0-100]
反馈：[详细反馈，包括优点和改进建议]
准确性总结：[总结当前答案的准确之处]
缺陷总结：[总结当前答案的主要缺陷]
自我反思：[回答上述两个反问]
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": get_system_prompt()},
                  {"role": "user", "content": judgment_prompt}]
    ).choices[0].message.content.strip()

    # 改进解析逻辑
    lines = response.split('\n')
    is_correct = False
    score = 0
    feedback = ""
    accuracy_summary = ""
    deficiency_summary = ""
    self_reflection = ""

    for line in lines:
        if line.startswith("是否正确："):
            is_correct = "是" in line
        elif line.startswith("得分："):
            try:
                score = float(line.split("：")[1].split("/")[0])
            except (ValueError, IndexError):
                score = 0
        elif line.startswith("反馈："):
            feedback = line.split("：", 1)[1].strip()
        elif line.startswith("准确性总结："):
            accuracy_summary = line.split("：", 1)[1].strip()
        elif line.startswith("缺陷总结："):
            deficiency_summary = line.split("：", 1)[1].strip()
        elif line.startswith("自我反思："):
            self_reflection = line.split("：", 1)[1].strip()
        else:
            # 将其他行添加到反馈中
            feedback += " " + line.strip()

    # 清理反馈中的方括号
    feedback = feedback.replace("[", "").replace("]", "").strip()

    # 改进错误处理逻辑
    if not feedback:
        feedback = "评判者未提供具体反馈。请检查答案的准确性和完整性。"
    if score == 0 and is_correct:
        score = 100  # 如果答案正确但没有给出分数，默认给满分
    elif score == 0 and not is_correct:
        score = 50  # 如果答案不正确且没有给出分数，默认给50分

    return JudgmentResult(
        is_correct=is_correct,
        score=score,
        feedback=feedback,
        use_web_info=True,
        accuracy_summary=accuracy_summary,
        deficiency_summary=deficiency_summary,
        self_reflection=self_reflection
    )

def display_reasoning_process(process: ReasoningProcess) -> None:
    """
    显示推理过程的详细信息。
    """
    bilingual_display("Problem:", "问题：")
    display(Markdown(f"{process.initial_problem}\n"))
    for idx, step in enumerate(process.steps, 1):
        bilingual_display(f"Step {idx}:", f"步骤 {idx}：")
        display(Markdown(f"**提示**：{step.hint}\n**是否完成**：{step.is_completed}\n"))
        if step.judgment:
            bilingual_display("Judgment Result:", "评判结：")
            display(Markdown(f"得分：{step.judgment.score}/100\n反馈：{step.judgment.feedback}\n使用网络信息：{'是' if step.judgment.use_web_info else '否'}\n"))
    if process.final_answer:
        bilingual_display("Final Answer:", "最终答案：")
        display(Markdown(f"{process.final_answer}"))
    else:
        bilingual_display("Final Answer:", "最终答案：")
        display(Markdown("尚未确定。"))
