#!/usr/bin/env python3
"""
测试本地Qwen2.5模型在GPQA数据集上的表现
整理模型答案和参考答案进行对比
"""

import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import re
import argparse

def load_gpqa_data(data_path):
    """加载GPQA数据"""
    df = pd.read_csv(data_path)
    
    # 选择需要的列
    questions = df['Question'].tolist()
    correct_answers = df['Correct Answer'].tolist()
    incorrect_answers_1 = df['Incorrect Answer 1'].tolist()
    incorrect_answers_2 = df['Incorrect Answer 2'].tolist()
    incorrect_answers_3 = df['Incorrect Answer 3'].tolist()
    
    # 创建选项列表
    options_list = []
    for i in range(len(questions)):
        options = [
            correct_answers[i],
            incorrect_answers_1[i],
            incorrect_answers_2[i],
            incorrect_answers_3[i]
        ]
        # 打乱选项顺序，但记住正确答案的位置
        options_list.append({
            'question': questions[i],
            'correct_answer': correct_answers[i],
            'options': options,
            'correct_index': 0  # 因为第一个选项是正确的
        })
    
    return options_list

def format_prompt(question, options):
    """格式化提示"""
    prompt = f"""请回答以下多项选择题。请仔细思考问题，并在最后给出你的答案。

问题：{question}

选项：
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

请将最终答案的选项字母包裹在`\\boxed{{}}`中。"""
    return prompt

def extract_model_answer(response):
    """从模型响应中提取答案"""
    match = re.search(r'\\boxed{([ABCD])}', response)
    if match:
        return match.group(1).upper()
    return None

def test_qwen_model(model_path, data_path):
    """测试Qwen模型"""
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载数据
    print("正在加载GPQA数据...")
    questions = load_gpqa_data(data_path)
    
    results = []
    
    print(f"开始测试 {len(questions)} 个问题...")
    for idx, item in enumerate(tqdm(questions)):
        prompt = format_prompt(item['question'], item['options'])
        
        # 生成回答
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # 提取模型答案
        model_answer = extract_model_answer(response)
        
        # 确定正确答案的字母
        correct_letter = chr(65 + item['correct_index'])  # 65是'A'的ASCII码
        
        result = {
            'question_id': idx + 1,
            'question': item['question'],
            'options': {
                'A': item['options'][0],
                'B': item['options'][1],
                'C': item['options'][2],
                'D': item['options'][3]
            },
            'correct_answer': correct_letter,
            'correct_text': item['correct_answer'],
            'model_response': response,
            'model_answer': model_answer,
            'is_correct': model_answer == correct_letter if model_answer else False
        }
        
        results.append(result)
        
        # 打印详细结果
        print(f"\n{'='*60}")
        print(f"问题 {idx + 1}:")
        print(f"问题：{item['question'][:100]}...")
        print(f"正确答案：{correct_letter}) {item['correct_answer']}")
        print(f"模型答案：{model_answer}")
        print(f"是否正确：{'✓' if result['is_correct'] else '✗'}")
        print(f"模型完整回答：{response[:200]}...")
    
    return results

def save_results(results, output_path):
    """保存测试结果"""
    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 生成汇总报告
    total_questions = len(results)
    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = correct_count / total_questions if total_questions > 0 else 0
    
    summary = {
        'total_questions': total_questions,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'detailed_results': results
    }
    
    summary_path = output_path.replace('.json', '_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试Qwen模型在GPQA上的表现")
    parser.add_argument("--model-path", default="./Qwen2.5-Math-1.5B", type=str)
    parser.add_argument("--data-path", default="./datasets/gpqa/gpqa_main.csv", type=str)
    parser.add_argument("--output-path", default="./qwen_gpqa_test_results.json", type=str)
    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    output_path = args.output_path
    
    if not os.path.exists(model_path):
        print(f"错误：模型路径 {model_path} 不存在")
        return
    
    if not os.path.exists(data_path):
        print(f"错误：数据文件 {data_path} 不存在")
        return
    
    # 运行测试
    results = test_qwen_model(model_path, data_path)
    
    # 保存结果
    summary = save_results(results, output_path)
    
    print(f"\n{'='*60}")
    print("测试完成！")
    print(f"总问题数：{summary['total_questions']}")
    print(f"正确数：{summary['correct_count']}")
    print(f"准确率：{summary['accuracy']:.2%}")
    print(f"详细结果已保存到：{output_path}")
    print(f"汇总报告已保存到：{output_path.replace('.json', '_summary.json')}")

if __name__ == "__main__":
    main()