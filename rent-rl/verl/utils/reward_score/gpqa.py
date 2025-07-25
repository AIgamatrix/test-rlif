import re

# Extraction Template from https://github.com/openai/simple-evals/blob/90e3e821cabba2aeb6be651dcb662b253df04225/common.py#L25
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-G])\$?"


def extract_solution(solution_str):
    match = re.search(ANSWER_PATTERN_MULTICHOICE, solution_str)
    return match.group(1) if match else None

def compute_score(solution_str, ground_truth) -> float:
    extracted_answer = extract_solution(solution_str)
    score = 1.0 if extracted_answer == ground_truth else 0.0
    return score