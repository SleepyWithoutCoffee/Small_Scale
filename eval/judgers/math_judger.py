from base_judger import Judger
import sys
import os
from jinja2 import Template

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.math.utils import extract_boxed_answer, patch_th_grade, patch_pm, grade_answer_mathd, grade_answer_sympy
from utils.llm_as_judge.api_router import Router

class MathJudger(Judger):
    def __init__(self, llm_as_judge=None):
        if llm_as_judge is not None:
            self.llm_as_judge = llm_as_judge
        super().__init__()

    def extract_answer(self, model_output: str) -> str:
        r"""Extract the answer from the passage."""
        if "\\boxed" in model_output:
            answer = extract_boxed_answer(model_output)
            answer = patch_th_grade(answer)
            answer = patch_pm(answer)
            return answer
        return None

    def _rule_based_grade_answer(self, given_answer: str, ground_truth: str):
        if grade_answer_mathd(given_answer, ground_truth):
            return True
        return grade_answer_sympy(given_answer, ground_truth)

    def _default_formatting(self, problem: str, given_answer: str, ground_truth: str):
        prompt_template = """You will be asked to look at the two answers (predicted and expected) to a math problem and to judge whether they are equivalent within the context of the problem.

Please first explain your reasoning in a couple of sentences. Then respond with only Yes or No as your judgement on whether the two answers are the same.
When comparing answers only perform trivial simplifications.

Here are a few examples.


{% raw %}Example 1:
Problem: Factor $7x^3 - 21x^2 + 14x$
Predicted answer: $7x(x - 2)(x - 1)$
Expected answer: $7x(x-1)(x-2)$

Reasoning: The order of the factors does not matter, so the answers are the same.
Judgement: Yes


Example 2:
Problem: A rectangle has a length of 6 meters and a width of 2 meters. If the length is reduced by 3 meters and the width is halved, what is the new area of the rectangle in square meters?
Predicted answer: 3/2
Expected answer: 1.5

Reasoning: 3/2 is the same as 1.5
Judgement: Yes


Example 3:
Problem: Simplify the expression $\\sqrt{7!}$, where $n!$ stands for $n \\cdot (n-1) \\cdot (n-2) \\cdots 2 \\cdot 1$.
Predicted answer: 71
Expected answer: 12\\sqrt{{35}}

Reasoning: This is non-trivial to simplify, so the answers are different.
Judgement: No


Example 4:
Problem: What is the simplified form of the expression $\\sqrt{98 x^{{3}} y^{{5}} z}}$?
\\begin{{align*}}
\\text{{A)}} & 2 x y z \\sqrt{{ 7 x y z}} &
\\text{{B)}} &  7 x^{{2}} y^{{2}} \\sqrt{{2 y z}}
\\\\
\\text{{C)}} & 7 x y^{{2}} \\sqrt{{2 x y z}}  &
\\text{{D)}} &49 x y^{{2}} \\sqrt{{2 x y z}}
\\end{{align*}}
Predicted answer: 7 x y^{{2}} \\sqrt{2xyz}$
Expected answer: $\\text{C}

Reasoning: Predicted answer is the same as the expected answer choice C.
Judgement: Yes


Example 5:
Problem: A line segment of length $5$ has one endpoint at $(1, 2)$ and the other endpoint at $(4, b)$. Find all possible values of $b$, separated by commas.
Predicted answer: -2, 6
Expected answer: 6, -2

Reasoning: The order doesn't matter in the context of the problem.
Judgement: Yes


Example 6:
Problem: Solve $\\tan x = \\sin x$ for $0 \\le x \\le 2\\pi$. Enter all the solutions, separated by commas.
Predicted answer: 0, \\pi
Expected answer: 0, \\pi, 2\\pi

Reasoning: Number of solutions is different.
Judgement: No



{% endraw %}YOUR TASK
Problem: {{problem}}
Predicted answer: {{predicted_answer}}
Expected answer: {{expected_answer}}
"""
        template = Template(prompt_template)
        content = template.render(problem=problem, predicted_answer=given_answer, expected_answer=ground_truth)
        return [
            {
                "role": "user",
                "content": content
        }]
        

    def grade_answer(self, given_answer: str, *, ground_truth: str = None, sample: dict = None):
        r"""Grade the answer."""
        if self.llm_as_judge is None or self.llm_as_judge["enable"] is False:
            return self._rule_based_grade_answer(given_answer, ground_truth)
        
        router = Router(self.llm_as_judge)
        # TODO: concurrent
        # TODO: original problem (not prompt)
        problem = sample["prompt"][0]["content"].replace("Please reason step by step, and put your final answer within \\boxed{}.", "")
        message = self._default_formatting(problem, given_answer, ground_truth)
        try:
            response = router.send_one_request(self.llm_as_judge["judge_model"]["model_name"], message, error_query_save_path=None)
            judgement = response.split("Judgement:")[-1].strip().lower()
            if judgement == "yes":
                result = True
            else:
                result = False
        except Exception as e:
            print(e)
            result = False
            
        return result
            




