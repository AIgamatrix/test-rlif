from datasets import load_dataset, Dataset
import huggingface_hub

# 登录Hugging Face（使用您的API令牌）
# huggingface_hub.login(token="您的API令牌")

# 加载原始数据集
ds = load_dataset("RLHFlow/numia_prompt_dpo1")

# 定义数据处理函数（增加索引参数idx）
def transform_data(example, idx):
    suffix = " Let's think step by step and output the final answer within \boxed{}."
    
    # 重构prompt格式
    new_prompt = [{
        "role": "user",
        "content": example["problem"] + suffix
    }]
    
    # 重构reward_model格式
    reward_model = {
        "ground_truth": example["gt"],
        "style": "rule"
    }
    
    # 创建extra_info字段
    extra_info = {
        "index": idx,  # 样本索引
        "name": "numina_math",  # 数据集名称
        "split": "train"  # 划分信息
    }
    
    return {
        "prompt": new_prompt,
        "reward_model": reward_model,
        "data_source": 'numina_math',
        "extra_info": extra_info
    }

# 应用数据处理，添加索引参数
formatted_ds = ds.map(transform_data, with_indices=True, remove_columns=["problem", "gt"])

# 推送处理后的数据集到Hub
formatted_ds.push_to_hub("qingyangzhang/numina_math_20K_formatted")