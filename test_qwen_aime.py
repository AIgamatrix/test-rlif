"""
测试本地Qwen2.5模型在AIME数据集上的表现
"""

import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import re
import argparse

def load_aime_data(data_path):
    """加载AIME数据"""
    df = pd.read_parquet(data_path)
    
    # 假设parquet文件包含 'problem' 和 'answer' 列
    questions = df['problem'].tolist()
    correct_answers = df['answer'].tolist()
    
    dataset = []
    for i in range(len(questions)):
        dataset.append({
            'question': questions[i],
            'correct_answer': correct_answers[i],
        })
    
    return dataset

def format_prompt(question):
    """格式化提示"""
    prompt = f"""请解决以下数学问题，并给出最终答案。请仔细思考，然后再末尾提供最终答案。
请将最终答案包裹在`\\boxed{{}}`中。
问题：{question}
"""
    return prompt

def extract_model_answer(response):
    """从模型响应中提取答案"""
    match = re.search(r'\\boxed{(.+?)}', response)
    if match:
        group_count = len(match.groups())
        # 最后一个分组的索引是 group_count
        last_group = match.group(group_count).strip()
        return last_group
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
    
    print("正在加载AIME数据...")
    dataset = load_aime_data(data_path)
    
    results = []
    
    print(f"开始测试 {len(dataset)} 个问题...")
    for idx, item in enumerate(tqdm(dataset)):
        prompt = format_prompt(item['question'])
        
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
        
        model_answer = extract_model_answer(response)
        
        is_correct = str(model_answer) == str(item['correct_answer']) if model_answer is not None else False
        
        result = {
            'question_id': idx + 1,
            'question': item['question'],
            'correct_answer': item['correct_answer'],
            'model_response': response,
            'model_answer': model_answer,
            'is_correct': is_correct
        }
        
        results.append(result)
        
        print(f"\n{'='*60}")
        print(f"问题 {idx + 1}:")
        print(f"问题：{item['question'][:100]}...")
        print(f"正确答案：{item['correct_answer']}")
        print(f"模型答案：{model_answer}")
        print(f"是否正确：{'✓' if result['is_correct'] else '✗'}")
        print(f"模型完整回答：{response}...")
    
    return results

def save_results(results, output_path):
    """保存测试结果"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
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
    parser = argparse.ArgumentParser(description="测试本地Qwen2.5模型在AIME数据集上的表现")
    parser.add_argument("--model-path", default="./Qwen2.5-Math-1.5B", type=str)
    parser.add_argument("--data-path", default="./datasets/aime24/data/train-00000-of-00001.parquet", type=str)
    parser.add_argument("--output-path", default="./qwen_aime_test_results.json", type=str)
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
    
    results = test_qwen_model(model_path, data_path)
    
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