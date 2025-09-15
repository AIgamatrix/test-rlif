#!/usr/bin/env python3
"""
验证转换后的HF模型能否正常加载和推理
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback

# 两个模型的HF转换路径（默认兜底，会被 --models 参数覆盖）
MODEL_PATHS = {
    "model_B_step2499": "/home/llama/test-rlif/checkpoints/EMPO/0824-052825/global_step_2499/actor_hf"
}

# 测试的提示语
TEST_PROMPTS = [
    "数学问题：小明有10个苹果，他吃了3个，请问还剩多少个？",
    "What is the capital of France?",
    "请解释一下机器学习的基本概念。"
]

def parse_args():
    parser = argparse.ArgumentParser(description="验证多个HF模型是否可加载与推理")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="模型列表，格式为 name:path。例如 --models EMPO:/path/to/hf TTRL:/path/to/hf"
    )
    return parser.parse_args()


def parse_models_arg(models_arg):
    if not models_arg:
        return MODEL_PATHS
    result = {}
    for item in models_arg:
        if ":" not in item:
            print(f"忽略无效模型参数: {item}（要求 name:path 格式）")
            continue
        name, path = item.split(":", 1)
        result[name.strip()] = path.strip()
    return result


def test_model(model_name, model_path):
    """测试单个模型的加载和推理"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"路径: {model_path}")
    print(f"{'='*60}")
    
    try:
        # 加载tokenizer和模型
        print("正在加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print("正在加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✓ 模型加载成功！")
        print(f"  - 模型类型: {type(model).__name__}")
        print(f"  - 参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - 词汇表大小: {len(tokenizer)}")
        
        # 测试推理
        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\n--- 测试 {i+1}: {prompt[:30]}... ---")
            try:
                # 编码输入
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                # 生成文本
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # 解码输出（仅解码新生成的部分）
                gen_part = outputs[0][inputs['input_ids'].shape[1]:]
                generated_text = tokenizer.decode(gen_part, skip_special_tokens=True)
                
                print(f"输入: {prompt}")
                print(f"输出: {generated_text.strip()}")
                print("✓ 推理成功！")
                
            except Exception as e:
                print(f"✗ 推理失败: {str(e)}")
                traceback.print_exc()
        
        # 清理内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ 模型加载失败: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """主函数"""
    args = parse_args()
    paths = parse_models_arg(args.models)

    print("开始验证转换后的HF模型...")
    
    success_count = 0
    total_count = len(paths)
    
    for model_name, model_path in paths.items():
        if test_model(model_name, model_path):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"验证完成！")
    print(f"成功: {success_count}/{total_count}")
    if success_count == total_count:
        print("✓ 所有模型都能正常加载和推理！")
    else:
        print(f"✗ {total_count - success_count} 个模型验证失败")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()