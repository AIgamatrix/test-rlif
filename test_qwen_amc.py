#!/usr/bin/env python3
"""
测试本地Qwen2.5模型在AMC12数据集上的表现
基于test_qwen_gpqa.py的风格，适配AMC12数据格式
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import argparse

class AMC12Tester:
    def __init__(self, model_path: str = "./Qwen2.5-Math-1.5B", data_path: str = "./datasets/amc12/data/train-00000-of-00001.parquet"):
        """初始化AMC12测试器"""
        self.model_path = model_path
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"正在加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载AMC12数据集
        self.load_amc12_data()
    
    def load_amc12_data(self):
        """加载AMC12数据集"""
        data_path = self.data_path
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"AMC12数据集未找到: {data_path}")
        
        print("正在加载AMC12数据集...")
        self.df = pd.read_parquet(data_path)
        print(f"数据集加载完成，共{len(self.df)}道题目")
        
        # 显示数据集基本信息
        print("\n数据集预览:")
        print(self.df.head())
    
    def format_prompt(self, problem: str) -> str:
        """格式化问题提示"""
        prompt = f"""请解决以下数学问题，并给出最终答案。请仔细思考，然后再末尾提供最终答案。
请将最终答案包裹在`\\boxed{{}}`中。
问题：{problem}。
"""
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return text
    
    def extract_answer(self, response: str) -> str:
        """从模型响应中提取答案"""
        match = re.search(r'\\boxed{(.+?)}', response)
        if match:
            group_count = len(match.groups())
            # 最后一个分组的索引是 group_count
            last_group = match.group(group_count).strip()
            return last_group
        return ""
    
    def run_single_test(self, problem: str, correct_answer: float) -> Dict[str, Any]:
        """运行单个测试"""
        prompt = self.format_prompt(problem)
        
        # 生成模型响应
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.6,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码响应
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        extracted_answer = self.extract_answer(response)
        
        # 检查答案是否正确
        try:
            # 尝试将提取的答案转换为浮点数
            pred_value = float(extracted_answer) if extracted_answer.replace('.', '', 1).isdigit() else None
            correct_value = float(correct_answer)
            is_correct = pred_value == correct_value if pred_value is not None else False
        except (ValueError, TypeError):
            is_correct = False
            pred_value = None
        
        return {
            "problem": problem,
            "correct_answer": correct_answer,
            "model_response": response,
            "extracted_answer": extracted_answer,
            "predicted_value": pred_value,
            "is_correct": is_correct
        }
    
    def run_tests(self) -> List[Dict[str, Any]]:
        """运行测试"""
        results = []
        
        sample_df = self.df
        
        print(f"\n开始测试，共{len(sample_df)}道题目...")
        
        for idx, row in sample_df.iterrows():
            print(f"\n测试题目 {len(results) + 1}/{len(sample_df)}")
            print(f"问题: {row['problem'][:100]}...")
            
            result = self.run_single_test(row['problem'], row['answer'])
            results.append(result)
            
            print(f"正确答案: {row['answer']}")
            print(f"模型答案: {result['extracted_answer']}")
            print(f"是否正确: {'✓' if result['is_correct'] else '✗'}")
            print(f"模型完整回答: {result['model_response']}")
            print(f"模型完整回答: {result['model_response']}")
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str = "qwen_amc12_test_results.json") -> Dict[str, Any]:
        """保存测试结果（与GPQA脚本风格保持一致）"""
        # 保存原始结果
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 生成汇总信息
        total_questions = len(results)
        correct_count = sum(1 for r in results if r["is_correct"])
        accuracy = correct_count / total_questions if total_questions else 0.0

        summary = {
            "model_path": self.model_path,
            "total_questions": total_questions,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "detailed_results": results,
        }

        summary_path = output_path.replace(".json", "_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n测试结果已保存到: {output_path}")
        print(f"汇总报告已保存到: {summary_path}")
        print(f"准确率: {accuracy:.2%} ({correct_count}/{total_questions})")

        return summary
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> str:
        """生成总结报告"""
        if not results:
            return "没有测试结果"
        
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        accuracy = correct / total
        
        report = f"""
# AMC12测试报告

## 测试概况
- **模型**: {self.model_path}
- **测试样本数**: {total}
- **正确数**: {correct}
- **准确率**: {accuracy:.2%}

## 详细结果
"""
        
        for i, result in enumerate(results, 1):
            status = "✓" if result['is_correct'] else "✗"
            report += f"""
### 题目 {i} {status}
- **问题**: {result['problem'][:200]}...
- **正确答案**: {result['correct_answer']}
- **模型答案**: {result['extracted_answer']}
- **是否正确**: {result['is_correct']}

"""
        
        return report

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AMC12 Qwen2.5模型测试")
    parser.add_argument("--model-path", default="./Qwen2.5-Math-1.5B", type=str)
    parser.add_argument("--data-path", default="./datasets/amc12/data/train-00000-of-00001.parquet", type=str)
    parser.add_argument("--output-path", default="./qwen_amc12_test_results.json", type=str)
    args = parser.parse_args()

    print("AMC12 Qwen2.5模型测试")
    print("=" * 50)
    
    # 初始化测试器
    tester = AMC12Tester(model_path=args.model_path, data_path=args.data_path)
    
    # 运行测试
    results = tester.run_tests()
    
    # 保存结果
    tester.save_results(results, args.output_path)
    
    # 生成报告
    report = tester.generate_summary_report(results)
    print("\n" + "=" * 50)
    print(report)

if __name__ == "__main__":
    main()