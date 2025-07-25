import argparse
import csv
import random
import os
from datasets import load_dataset

def main(output_path):
    # Load the DeepMind CodeContest dataset from HuggingFace
    dataset = load_dataset("deepmind/code_contests", split="train")
    print(f"Loaded {len(dataset)} examples from the dataset.")

    # Define prompt templates suitable for reinforcement learning
    prompt_templates = [
        # Template 1
        ("You are an expert Python programmer. You will be given a question (problem specification) "
         "and will generate a correct Python program that meets the requirements and passes all tests. "
         "Do not include any explanations or outputs; return only the final code.\n\n"
         "### Question:\n{description}\n\n"
         "### Answer:\n```python\n# Your code here\n```"),
        # Template 2
        ("Solve the following problem by writing a correct and efficient Python program. "
         "The output should be just the code solution.\n\n"
         "Problem:\n{description}\n\n"
         "Solution (in Python code):\n```python\n# Your code here\n```"),
        # Template 3
        ("Write a Python function or program that fulfills the task described below. "
         "Ensure the solution is correct and handles all edge cases. Provide only the code as output.\n\n"
         "Task Description:\n{description}\n\n"
         "Complete Solution:\n```python\n# Your code here\n```"),
        # Template 4
        ("{description}\n\n"
         "Provide a Python program that solves the above problem. Output only the code, nothing else:\n```python\n# Your code here\n```")
    ]

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Open CSV for writing
    with open(output_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["task_id", "prompt"])

        # Iterate through the dataset
        for example in dataset:
            task_id = example.get("task_id", "")
            description = example.get("description", "").strip()

            # Skip if description is empty
            if not description:
                continue

            # Select a prompt template at random
            template = random.choice(prompt_templates)
            prompt = template.format(description=description)

            # Write to CSV
            writer.writerow([task_id, prompt])

    print(f"Finished generating prompts. Output written to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts from the DeepMind CodeContest dataset.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="~/data/codecontest/train.csv",
        help="Path to the output CSV file (default: ~/data/codecontest/train.csv)"
    )
    args = parser.parse_args()
    main(args.output_path)
