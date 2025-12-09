#!/usr/bin/env python3
"""
图片 Caption 重试脚本

该脚本用于重新生成之前失败的图片caption。
使用阿里云 DashScope API (qwen3-vl-plus) 模型生成图片描述。

特点：
- 只处理caption为空的图片
- 支持本地和远程图片
- 使用流式API调用
"""

import json
import os
import argparse
import logging
import base64
import hashlib
import requests
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import time
from openai import OpenAI

# 禁用SSL警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# DashScope API配置
DASHSCOPE_API_KEY = "sk-7a4e2ece8871495895a9c6a506715e9b"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-vl-plus"

# 支持的图片格式
SUPPORTED_IMAGE_FORMATS = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp'
}


def setup_logging(level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('retry_caption_generation.log')
        ]
    )


def download_remote_image(url: str, temp_dir: str) -> Optional[str]:
    """
    下载远程图片到临时目录
    
    Args:
        url: 图片URL
        temp_dir: 临时目录路径
        
    Returns:
        str: 下载后的本地路径，失败返回None
    """
    try:
        # 从URL中提取文件名
        url_path = url.split('?')[0]  # 去掉查询参数
        filename = os.path.basename(url_path)
        if not filename:
            filename = hashlib.md5(url.encode()).hexdigest()
        
        # 确保有正确的扩展名
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_IMAGE_FORMATS:
            filename = f"{filename}.png"
        
        local_path = os.path.join(temp_dir, filename)
        
        # 下载图片
        response = requests.get(url, timeout=30, verify=False)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return local_path
        else:
            logging.warning(f"下载远程图片失败 {url}: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        logging.warning(f"下载远程图片异常 {url}: {str(e)}")
        return None


def encode_image_to_base64(image_path: str) -> Optional[Tuple[str, str]]:
    """
    将图片转换为Base64编码
    
    Args:
        image_path: 图片路径
        
    Returns:
        Tuple[str, str]: (base64编码, MIME类型) 或 None（如果失败）
    """
    try:
        if not os.path.exists(image_path):
            return None
            
        # 获取文件扩展名
        ext = Path(image_path).suffix.lower()
        if ext not in SUPPORTED_IMAGE_FORMATS:
            return None
            
        mime_type = SUPPORTED_IMAGE_FORMATS[ext]
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
        return base64_image, mime_type
        
    except Exception as e:
        logging.warning(f"编码图片失败 {image_path}: {str(e)}")
        return None


def generate_caption_dashscope(image_path: str, context: str = "", 
                                api_key: str = DASHSCOPE_API_KEY,
                                max_retries: int = 3) -> Optional[str]:
    """
    使用DashScope API生成图片caption
    
    Args:
        image_path: 图片路径（本地路径）
        context: 图片的上下文信息
        api_key: API密钥
        max_retries: 最大重试次数
        
    Returns:
        str: 生成的caption，失败返回None
    """
    # 编码图片
    result = encode_image_to_base64(image_path)
    if result is None:
        logging.warning(f"无法编码图片: {image_path}")
        return None
        
    base64_image, mime_type = result
    
    # 构建提示词
    prompt = "请用约100字简要描述这张图片的内容。"
    if context:
        prompt = f"这张图片出现在以下文档上下文中：\n{context[:400]}\n\n请结合上下文，用约100字简要描述这张图片展示的核心内容。直接描述，不要分点列举。"
    
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL
    )
    
    for attempt in range(max_retries):
        try:
            # 创建聊天完成请求（流式）
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                stream=True,
                extra_body={
                    'enable_thinking': False,  # 不需要思考过程
                },
            )
            
            # 收集回复内容
            answer_content = ""
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        answer_content += delta.content
            
            if answer_content.strip():
                return answer_content.strip()
            else:
                logging.warning(f"API返回空内容 [{image_path}] (尝试 {attempt + 1}/{max_retries})")
                
        except Exception as e:
            logging.warning(f"API调用异常 [{image_path}] (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            
        if attempt < max_retries - 1:
            time.sleep(2)  # 重试前等待2秒
    
    logging.error(f"图片caption生成失败 [{image_path}]: 已重试 {max_retries} 次")
    return None


class FailedCaptionRetryProcessor:
    """
    失败Caption重试处理器
    
    负责找出caption为空的图片，重新生成caption
    """
    
    def __init__(self, api_key: str = DASHSCOPE_API_KEY):
        """
        初始化处理器
        
        Args:
            api_key: API密钥
        """
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # 临时目录（用于存储下载的远程图片）
        self.temp_dir = None
        
        # 统计信息
        self.stats = {
            'total_images': 0,
            'empty_captions': 0,
            'api_calls': 0,
            'api_success': 0,
            'failed': 0,
            'skipped': 0
        }
        
    def _collect_empty_caption_images(self, doc_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        收集所有caption为空的图片
        
        Args:
            doc_structure: 文档结构
            
        Returns:
            List: 需要重试的图片任务列表
        """
        image_tasks = []
        
        def collect_recursive(obj: Any, path: str = ""):
            if isinstance(obj, dict):
                if 'images' in obj and isinstance(obj['images'], dict):
                    for image_id, image_info in obj['images'].items():
                        if isinstance(image_info, dict):
                            self.stats['total_images'] += 1
                            
                            # 检查caption是否为空
                            caption = image_info.get('caption', '')
                            if not caption or caption.strip() == '':
                                abs_path = image_info.get('absolute_path', '')
                                is_remote = image_info.get('is_remote', False)
                                
                                if abs_path:
                                    self.stats['empty_captions'] += 1
                                    image_tasks.append({
                                        'path': path,
                                        'image_id': image_id,
                                        'absolute_path': abs_path,
                                        'is_remote': is_remote,
                                        'context': image_info.get('context', ''),
                                        'info': image_info
                                    })
                                
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    collect_recursive(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    collect_recursive(item, f"{path}[{i}]")
                    
        collect_recursive(doc_structure)
        
        return image_tasks
        
    def _generate_caption_for_image(self, task: Dict[str, Any]) -> Tuple[str, str]:
        """
        为单个图片任务生成caption
        
        Args:
            task: 图片任务信息
            
        Returns:
            Tuple[str, str]: (image_id, caption)
        """
        image_id = task['image_id']
        abs_path = task['absolute_path']
        is_remote = task['is_remote']
        context = task['context']
        
        # 确定实际的图片路径
        actual_path = abs_path
        
        if is_remote:
            # 远程图片：下载到临时目录
            if self.temp_dir:
                downloaded_path = download_remote_image(abs_path, self.temp_dir)
                if downloaded_path:
                    actual_path = downloaded_path
                else:
                    self.stats['failed'] += 1
                    return image_id, ""
            else:
                self.stats['failed'] += 1
                return image_id, ""
        else:
            # 本地图片：检查是否存在
            if not os.path.exists(abs_path):
                self.logger.warning(f"本地图片不存在: {abs_path}")
                self.stats['failed'] += 1
                return image_id, ""
            
        # 调用API生成caption
        self.stats['api_calls'] += 1
        caption = generate_caption_dashscope(actual_path, context, self.api_key)
        
        if caption:
            self.stats['api_success'] += 1
            return image_id, caption
        else:
            self.stats['failed'] += 1
            return image_id, ""
            
    def process_document_structure(self, doc_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理文档结构，为所有空caption的图片重新生成caption
        
        Args:
            doc_structure: 文档结构
            
        Returns:
            Dict: 更新后的文档结构
        """
        import copy
        updated_structure = copy.deepcopy(doc_structure)
        
        # 创建临时目录（用于远程图片）
        self.temp_dir = tempfile.mkdtemp(prefix='retry_caption_')
        self.logger.info(f"创建临时目录: {self.temp_dir}")
        
        try:
            # 收集所有空caption的图片
            self.logger.info("收集空caption图片信息...")
            image_tasks = self._collect_empty_caption_images(updated_structure)
            
            self.logger.info(f"总图片数: {self.stats['total_images']}")
            self.logger.info(f"空caption图片数: {self.stats['empty_captions']}")
            
            if not image_tasks:
                self.logger.info("没有找到需要重试的图片")
                return updated_structure
                
            # 生成caption
            caption_map = {}  # image_id -> caption
            
            for task in tqdm(image_tasks, desc="重新生成图片caption"):
                image_id, caption = self._generate_caption_for_image(task)
                if caption:
                    caption_map[image_id] = caption
                # 添加延迟避免API限流
                time.sleep(0.5)
                    
            # 更新结构中的caption
            self.logger.info("更新文档结构中的caption...")
            self._update_captions_in_structure(updated_structure, caption_map)
            
        finally:
            # 清理临时目录
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"清理临时目录: {self.temp_dir}")
        
        return updated_structure
        
    def _update_captions_in_structure(self, obj: Any, caption_map: Dict[str, str]):
        """
        递归更新结构中所有图片的caption
        
        Args:
            obj: 要更新的对象
            caption_map: image_id -> caption 的映射
        """
        if isinstance(obj, dict):
            if 'images' in obj and isinstance(obj['images'], dict):
                for image_id, image_info in obj['images'].items():
                    if isinstance(image_info, dict):
                        if image_id in caption_map:
                            image_info['caption'] = caption_map[image_id]
                            
            for value in obj.values():
                self._update_captions_in_structure(value, caption_map)
        elif isinstance(obj, list):
            for item in obj:
                self._update_captions_in_structure(item, caption_map)
                
    def save_updated_structure(self, updated_structure: Dict[str, Any], output_file: str):
        """
        保存更新后的文档结构
        
        Args:
            updated_structure: 更新后的文档结构
            output_file: 输出文件路径
        """
        try:
            output_dir = Path(output_file).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(updated_structure, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"更新后的结构已保存到: {output_file}")
            
        except Exception as e:
            self.logger.error(f"保存文件时出错: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="重新生成失败的图片caption（使用DashScope API）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python retry_failed_captions.py input.json output.json
  python retry_failed_captions.py input.json output.json --dry-run
  python retry_failed_captions.py input.json output.json --api-key YOUR_KEY
        """
    )
    
    parser.add_argument(
        "input_file",
        help="输入的文档结构JSON文件路径（包含已生成的caption）"
    )
    
    parser.add_argument(
        "output_file",
        help="输出的文档结构JSON文件路径"
    )
    
    parser.add_argument(
        "--api-key",
        default=DASHSCOPE_API_KEY,
        help="DashScope API密钥"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只分析空caption图片数量，不实际生成"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志输出"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    
    # 验证输入文件
    if not os.path.exists(args.input_file):
        logger.error(f"输入文件不存在: {args.input_file}")
        return 1
        
    # 加载文档结构
    logger.info(f"加载文档结构: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        doc_structure = json.load(f)
        
    logger.info(f"文档结构包含 {len(doc_structure)} 个文件")
    
    # 创建处理器
    processor = FailedCaptionRetryProcessor(api_key=args.api_key)
    
    if args.dry_run:
        # 只分析空caption图片
        logger.info("=== 执行预分析 ===")
        image_tasks = processor._collect_empty_caption_images(doc_structure)
        
        logger.info(f"图片统计:")
        logger.info(f"  总图片数: {processor.stats['total_images']}")
        logger.info(f"  空caption图片数: {processor.stats['empty_captions']}")
        
        # 估算时间（每个图片约2秒）
        estimated_time = processor.stats['empty_captions'] * 2
        logger.info(f"  预估处理时间: {estimated_time} 秒 ({estimated_time/60:.1f} 分钟)")
        
        # 列出前10个需要重试的图片
        if image_tasks:
            logger.info(f"\n前10个需要重试的图片:")
            for task in image_tasks[:10]:
                logger.info(f"  - {task['image_id']}: {task['absolute_path']}")
        
        return 0
        
    # 处理文档结构
    logger.info("开始重新生成失败的图片caption...")
    try:
        updated_structure = processor.process_document_structure(doc_structure)
        
        # 保存结果
        logger.info(f"保存结果到: {args.output_file}")
        processor.save_updated_structure(updated_structure, args.output_file)
        
        logger.info("图片caption重试完成！")
        
        # 输出统计
        stats = processor.stats
        logger.info(f"处理统计:")
        logger.info(f"  总图片数: {stats['total_images']}")
        logger.info(f"  空caption图片数: {stats['empty_captions']}")
        logger.info(f"  API调用次数: {stats['api_calls']}")
        logger.info(f"  API成功次数: {stats['api_success']}")
        logger.info(f"  失败: {stats['failed']}")
        
        if stats['empty_captions'] > 0:
            success_rate = stats['api_success'] / stats['empty_captions'] * 100
            logger.info(f"  成功率: {success_rate:.1f}%")
            
    except KeyboardInterrupt:
        logger.warning("用户中断了处理过程")
        return 1
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())




