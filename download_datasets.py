#!/usr/bin/env python3
"""
数据集下载脚本
使用 Hugging Face 镜像下载数学相关数据集
"""

import os
import argparse
from huggingface_hub import snapshot_download
from typing import List, Dict

# 数据集配置
DATASET_CONFIGS = {
    "math": {
        "repo_id": "hendrycks/competition_math",
        "description": "MATH dataset - 数学竞赛问题数据集",
        "repo_type": "dataset"
    },
    "aime24": {
        "repo_id": "AI-MO/aime_2024", 
        "description": "AIME 2024 dataset - 2024年美国数学邀请赛",
        "repo_type": "dataset"
    },
    "gpqa": {
        "repo_id": "Idavidrein/gpqa",
        "description": "GPQA dataset - 研究生级别物理、化学、生物问答",
        "repo_type": "dataset"
    },
    "minerva": {
        "repo_id": "math-ai/minervamath",
        "description": "Minerva Math dataset - 数学推理数据集",
        "repo_type": "dataset"
    },
    "amc12": {
        "repo_id": "AI-MO/aimo-validation-amc",
        "description": "AMC 12 dataset - 美国数学竞赛12年级",
        "repo_type": "dataset"
    },
}

class DatasetDownloader:
    def __init__(self, mirror_endpoint: str = "https://hf-mirror.com", base_dir: str = "./datasets", token: str = None):
        """
        初始化数据集下载器
        
        Args:
            mirror_endpoint: Hugging Face 镜像地址
            base_dir: 数据集保存的基础目录
            token: Hugging Face API token，用于下载需要登录的数据集
        """
        self.mirror_endpoint = mirror_endpoint
        self.base_dir = base_dir
        self.token = token
        
        # 确保基础目录存在
        os.makedirs(self.base_dir, exist_ok=True)
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> str:
        """
        下载指定数据集
        
        Args:
            dataset_name: 数据集名称
            force_download: 是否强制重新下载
            
        Returns:
            str: 下载后的本地路径
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
        
        config = DATASET_CONFIGS[dataset_name]
        repo_id = config["repo_id"]
        local_dir = os.path.join(self.base_dir, dataset_name)
        
        print(f"\n{'='*60}")
        print(f"Downloading {dataset_name}")
        print(f"Description: {config['description']}")
        print(f"Repository: {repo_id}")
        print(f"Local directory: {local_dir}")
        print(f"Mirror endpoint: {self.mirror_endpoint}")
        print(f"{'='*60}")
        
        # 检查是否已存在
        if os.path.exists(local_dir) and not force_download:
            print(f"Dataset {dataset_name} already exists at {local_dir}")
            print("Use --force to re-download")
            return local_dir
        
        try:
            # 使用 snapshot_download 下载数据集
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                endpoint=self.mirror_endpoint,
                resume_download=True,  # 支持断点续传
                local_dir_use_symlinks=False,  # 不使用符号链接
                repo_type="dataset",  # 明确指定为数据集类型
                token=self.token if self.token else None
            )
            
            print(f"✅ Successfully downloaded {dataset_name} to {local_dir}")
            
            # 显示下载的文件结构
            self._show_directory_structure(local_dir)
            
            return downloaded_path
            
        except Exception as e:
            print(f"❌ Failed to download {dataset_name}: {str(e)}")
            raise
    
    def download_multiple_datasets(self, dataset_names: List[str], force_download: bool = False) -> Dict[str, str]:
        """
        下载多个数据集
        
        Args:
            dataset_names: 数据集名称列表
            force_download: 是否强制重新下载
            
        Returns:
            Dict[str, str]: 数据集名称到本地路径的映射
        """
        results = {}
        failed_downloads = []
        
        for dataset_name in dataset_names:
            try:
                local_path = self.download_dataset(dataset_name, force_download)
                results[dataset_name] = local_path
            except Exception as e:
                print(f"Failed to download {dataset_name}: {str(e)}")
                failed_downloads.append(dataset_name)
        
        # 总结下载结果
        print(f"\n{'='*60}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successfully downloaded: {len(results)} datasets")
        for name, path in results.items():
            print(f"  - {name}: {path}")
        
        if failed_downloads:
            print(f"❌ Failed downloads: {len(failed_downloads)} datasets")
            for name in failed_downloads:
                print(f"  - {name}")
        
        return results
    
    def _show_directory_structure(self, directory: str, max_files: int = 10):
        """
        显示目录结构
        
        Args:
            directory: 目录路径
            max_files: 最多显示的文件数量
        """
        print(f"\nDirectory structure of {directory}:")
        try:
            items = os.listdir(directory)
            items.sort()
            
            for i, item in enumerate(items[:max_files]):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                else:
                    size = os.path.getsize(item_path)
                    size_str = self._format_size(size)
                    print(f"  📄 {item} ({size_str})")
            
            if len(items) > max_files:
                print(f"  ... and {len(items) - max_files} more items")
                
        except Exception as e:
            print(f"  Error reading directory: {e}")
    
    def _format_size(self, size_bytes: int) -> str:
        """
        格式化文件大小
        
        Args:
            size_bytes: 字节大小
            
        Returns:
            str: 格式化后的大小字符串
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def list_available_datasets(self):
        """
        列出所有可用的数据集
        """
        print("Available datasets:")
        print("=" * 60)
        for name, config in DATASET_CONFIGS.items():
            print(f"📊 {name}")
            print(f"   Repository: {config['repo_id']}")
            print(f"   Description: {config['description']}")
            print()

def main():
    parser = argparse.ArgumentParser(
        description="Download math datasets using Hugging Face mirror",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download single dataset
  python download_datasets.py --datasets math
  
  # Download multiple datasets
  python download_datasets.py --datasets math aime24 gpqa
  
  # Download all datasets
  python download_datasets.py --datasets all
  
  # List available datasets
  python download_datasets.py --list
  
  # Use custom mirror and directory
  python download_datasets.py --datasets math --mirror https://hf-mirror.com --output-dir ./my_datasets
        """
    )
    
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        help="Dataset names to download (use 'all' for all datasets)"
    )
    parser.add_argument(
        "--output-dir", 
        default="./datasets",
        help="Output directory for datasets (default: ./datasets)"
    )
    parser.add_argument(
        "--mirror", 
        default="https://hf-mirror.com",
        help="Hugging Face mirror endpoint (default: https://hf-mirror.com)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-download even if dataset exists"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List available datasets"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token for downloading restricted datasets"
    )
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = DatasetDownloader(
        mirror_endpoint=args.mirror,
        base_dir=args.output_dir,
        token=args.token
    )
    
    # 列出可用数据集
    if args.list:
        downloader.list_available_datasets()
        return
    
    # 检查是否提供了数据集参数
    if not args.datasets:
        print("Error: Please specify datasets to download or use --list to see available datasets")
        parser.print_help()
        return
    
    # 处理 'all' 选项
    if "all" in args.datasets:
        dataset_names = list(DATASET_CONFIGS.keys())
    else:
        dataset_names = args.datasets
    
    # 验证数据集名称
    invalid_datasets = [name for name in dataset_names if name not in DATASET_CONFIGS]
    if invalid_datasets:
        print(f"Error: Unknown datasets: {invalid_datasets}")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        return
    
    # 下载数据集
    try:
        results = downloader.download_multiple_datasets(dataset_names, args.force)
        
        if results:
            print(f"\n🎉 Download completed! Datasets saved to: {args.output_dir}")
        else:
            print("\n❌ No datasets were successfully downloaded")
            
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Download failed: {str(e)}")

if __name__ == "__main__":
    main()