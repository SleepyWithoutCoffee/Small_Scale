import re
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base_judger import Judger
from utils.mcq.utils import extract_boxed_answer
NUMBER_OF_CHOICES = 4

class McqJudger(Judger):
    def extract_answer(self, model_output: str) -> str:
        r"""Extract the answer from the passage."""
        if "\\boxed" in model_output:
            answer = extract_boxed_answer(model_output)
        else:
            END_LETTER = chr(ord('A') + NUMBER_OF_CHOICES - 1)
            pattern_1 = r"(?i)Answer[ \t]*:[ \t]*\$?([A-{END_LETTER}])\$?"
            pattern_2 = r"answer is \(?([A-{END_LETTER}])\)?"
            matches_1 = list(re.finditer(pattern_1, model_output))
            if matches_1:
                answer = matches_1[-1].group(1)
            else:
                matches_2 = list(re.finditer(pattern_2, model_output))
                if matches_2:
                    answer = matches_2[-1].group(1)
                else:
                    pattern = r"\b[A-{END_LETTER}]\b(?!.*\b[A-{END_LETTER}]\b)"
                    match = re.search(pattern, model_output, re.DOTALL)
                    if match:
                        answer = match.group(0)
                    else:
                        answer = None
                # answer = None
        return answer


    def grade_answer(self, given_answer: str, *, ground_truth: str = None, sample: dict = None):
        if given_answer is None:
            return False
        ground_truth = ground_truth.lower()
        given_answer = given_answer.lower()
        return given_answer in ground_truth
    
