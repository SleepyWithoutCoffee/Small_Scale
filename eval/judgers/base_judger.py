from abc import ABC, abstractmethod

class Judger(ABC):
    @abstractmethod
    def extract_answer(self, model_output: str) -> str:
        pass

    @abstractmethod
    def grade_answer(self, given_answer: str, *, ground_truth: str = None, sample: dict = None):
        pass

    def test_equal(self, response, ground_truth, extract_from_ground_truth=False, has_prediction=False, ignore_think_token=False):
        if ignore_think_token and "</think>" in response:
            response = response.split("</think>")[1].strip()
        if extract_from_ground_truth:
            ground_truth = self.extract_answer(ground_truth)
        if has_prediction:
            answer = response
        else:
            answer = self.extract_answer(response)
        is_correct = self.grade_answer(answer, ground_truth=ground_truth)
        return is_correct