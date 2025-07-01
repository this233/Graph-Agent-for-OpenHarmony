import re
import hashlib
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import argparse


class MarkdownParser:
    def __init__(self):
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```[\s\S]*?```', re.DOTALL)
        self.table_pattern = re.compile(r'^\|.*?\|$(?:\n^\|.*?\|$)*', re.MULTILINE)
        self.link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        
    def generate_id(self, content: str, prefix: str) -> str:
        """生成基于内容MD5的ID"""
        md5_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{prefix}-{md5_hash}"
    
    def add_heading_numbers(self, content: str) -> str:
        """给标题添加分层序号"""
        lines = content.split('\n')
        result = []
        heading_counters = [0] * 6  # 支持6级标题
        
        for line in lines:
            heading_match = self.heading_pattern.match(line)
            if heading_match:
                level = len(heading_match.group(1))  # 标题级别
                title = heading_match.group(2).strip()
                
                # 检查是否已经有序号
                if re.match(r'^\d+(\.\d+)*\.?\s', title):
                    result.append(line)
                    continue
                
                # 更新计数器
                heading_counters[level - 1] += 1
                # 重置下级计数器
                for i in range(level, 6):
                    heading_counters[i] = 0
                
                # 生成序号
                number_parts = []
                for i in range(level):
                    if heading_counters[i] > 0:
                        number_parts.append(str(heading_counters[i]))
                
                number_str = '.'.join(number_parts)
                numbered_title = f"{number_str}. {title}"
                
                result.append(f"{'#' * level} {numbered_title}")
            else:
                result.append(line)
        
        return '\n'.join(result)
    
    def extract_headings(self, content: str) -> List[Tuple[int, str, int]]:
        """提取所有标题，返回 (level, title, start_pos)"""
        headings = []
        for match in self.heading_pattern.finditer(content):
            level = len(match.group(1))
            title = match.group(2).strip()
            start_pos = match.start()
            headings.append((level, title, start_pos))
        return headings
    
    def extract_code_blocks(self, content: str) -> Dict[str, Dict[str, str]]:
        """提取代码块"""
        codes = {}
        for match in self.code_block_pattern.finditer(content):
            code_content = match.group(0)
            code_id = self.generate_id(code_content, "code")
            codes[code_id] = {
                "abstract": "",
                "content": code_content
            }
        return codes
    
    def extract_tables(self, content: str) -> Dict[str, Dict[str, str]]:
        """提取表格"""
        tables = {}
        # 改进的表格匹配模式
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('|') and line.endswith('|'):
                # 找到表格开始
                table_lines = [lines[i]]
                j = i + 1
                while j < len(lines) and lines[j].strip().startswith('|') and lines[j].strip().endswith('|'):
                    table_lines.append(lines[j])
                    j += 1
                
                if len(table_lines) >= 2:  # 至少有标题行和分隔行
                    table_content = '\n'.join(table_lines)
                    table_id = self.generate_id(table_content, "table")
                    tables[table_id] = {
                        "abstract": "",
                        "content": table_content
                    }
                
                i = j
            else:
                i += 1
        
        return tables
    
    def extract_links(self, content: str) -> Dict[str, Dict[str, str]]:
        """提取超链接"""
        jumps = {}
        for match in self.link_pattern.finditer(content):
            link_text = match.group(1)
            link_url = match.group(2)
            
            # 只处理相对路径的文件链接
            if not link_url.startswith(('http://', 'https://', 'mailto:', '#')):
                file_id = self.generate_id(link_url, "file")
                jumps[file_id] = {
                    "file_path": link_url
                }
        return jumps
    
    def remove_code_and_tables(self, content: str) -> str:
        """移除代码块和表格"""
        # 移除代码块
        content = self.code_block_pattern.sub('', content)
        
        # 移除表格
        lines = content.split('\n')
        result_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('|') and line.endswith('|'):
                # 跳过表格行
                while i < len(lines) and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                    i += 1
            else:
                result_lines.append(lines[i])
                i += 1
        
        content = '\n'.join(result_lines)
        # 清理多余的空行
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        return content.strip()
    
    def get_heading_hierarchy(self, headings: List[Tuple[int, str, int]], current_index: int) -> List[str]:
        """获取当前标题的层级路径"""
        if current_index >= len(headings):
            return []
        
        current_level = headings[current_index][0]
        hierarchy = [headings[current_index][1]]
        
        # 向上查找父级标题
        for i in range(current_index - 1, -1, -1):
            level, title, _ = headings[i]
            if level < current_level:
                hierarchy.insert(0, title)
                current_level = level
        
        return hierarchy
    
    def split_content_by_headings(self, content: str, target_level: int) -> List[Tuple[str, str, int, int]]:
        """按指定级别的标题分割内容，返回 (chunk_content, chunk_title, start_pos, end_pos)"""
        headings = self.extract_headings(content)
        if not headings:
            return [(content, "", 0, len(content))]
        
        # 找到目标级别的标题
        target_headings = []
        for i, (level, title, start_pos) in enumerate(headings):
            if level == target_level:
                target_headings.append((i, level, title, start_pos))
        
        if not target_headings:
            return [(content, "", 0, len(content))]
        
        chunks = []
        content_lines = content.split('\n')
        
        for i, (heading_index, level, title, start_pos) in enumerate(target_headings):
            # 找到标题在文本中的行号
            heading_line = None
            for line_num, line in enumerate(content_lines):
                if f"{'#' * level} {title}" == line.strip():
                    heading_line = line_num
                    break
            
            if heading_line is None:
                continue
            
            # 找到chunk的结束位置
            end_line = len(content_lines)
            if i + 1 < len(target_headings):
                next_heading_index, next_level, next_title, next_start_pos = target_headings[i + 1]
                for line_num, line in enumerate(content_lines[heading_line + 1:], heading_line + 1):
                    if f"{'#' * next_level} {next_title}" == line.strip():
                        end_line = line_num
                        break
            
            # 获取chunk内容
            chunk_lines = content_lines[heading_line:end_line]
            
            # 添加父级标题
            hierarchy = self.get_heading_hierarchy(headings, heading_index)
            parent_titles = hierarchy[:-1]  # 除了当前标题
            
            # 构建包含层级标题的chunk内容
            chunk_content_lines = []
            for parent_title in parent_titles:
                # 找到父级标题的原始格式
                for h_level, h_title, _ in headings:
                    if h_title == parent_title:
                        chunk_content_lines.append(f"{'#' * h_level} {h_title}")
                        break
            
            chunk_content_lines.extend(chunk_lines)
            
            # 移除子级标题的内容
            filtered_lines = []
            skip_mode = False
            for line in chunk_content_lines:
                heading_match = self.heading_pattern.match(line)
                if heading_match:
                    line_level = len(heading_match.group(1))
                    if line_level > level:
                        skip_mode = True
                        continue
                    else:
                        skip_mode = False
                
                if not skip_mode:
                    filtered_lines.append(line)
            
            chunk_content = '\n'.join(filtered_lines)
            chunks.append((chunk_content, title, heading_line, end_line))
        
        return chunks
    
    def process_chunk(self, content: str, title: str, level: int = 1) -> Dict[str, Any]:
        """处理单个chunk"""
        # 提取各种元素
        codes = self.extract_code_blocks(content)
        tables = self.extract_tables(content)
        jumps = self.extract_links(content)
        
        # 创建filter_chunk
        filtered_content = self.remove_code_and_tables(content)
        filter_chunk = {
            "content": filtered_content,
            "extracted_entities": "",  # 占位符
            "extracted_triples": ""    # 占位符
        }
        
        # 创建chunk结构
        chunk_data = {
            "abstract": "",  # 占位符
            "content": content,
            "jump": jumps,
            "codes": codes,
            "tables": tables,
            "filter_chunk": filter_chunk
        }
        
        # 递归处理子级chunks
        sub_chunks = self.split_content_by_headings(content, level + 1)
        if len(sub_chunks) > 1 or (len(sub_chunks) == 1 and sub_chunks[0][1] != ""):
            chunk_data["chunks"] = {}
            for sub_content, sub_title, _, _ in sub_chunks:
                if sub_title:  # 只处理有标题的chunk
                    sub_chunk_data = self.process_chunk(sub_content, sub_title, level + 1)
                    sub_chunk_id = self.generate_id(sub_content, "chunk")
                    chunk_data["chunks"][sub_chunk_id] = sub_chunk_data
        
        return chunk_data
    
    def parse_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """解析Markdown文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
        
        # 添加标题序号
        numbered_content = self.add_heading_numbers(content)
        
        # 生成文件ID
        file_id = self.generate_id(numbered_content, "file")
        
        # 创建文件结构
        file_data = {
            "abstract": "",  # 占位符
            "file_path": file_path,
            "content": numbered_content,
            "chunks": {}
        }
        
        # 按一级标题分割
        level_1_chunks = self.split_content_by_headings(numbered_content, 1)
        
        if len(level_1_chunks) > 1 or (len(level_1_chunks) == 1 and level_1_chunks[0][1] != ""):
            for chunk_content, chunk_title, _, _ in level_1_chunks:
                if chunk_title:  # 只处理有标题的chunk
                    chunk_data = self.process_chunk(chunk_content, chunk_title, 1)
                    chunk_id = self.generate_id(chunk_content, "chunk")
                    file_data["chunks"][chunk_id] = chunk_data
        else:
            # 如果没有一级标题，整个文档作为一个chunk
            chunk_data = self.process_chunk(numbered_content, "Main Content", 1)
            chunk_id = self.generate_id(numbered_content, "chunk")
            file_data["chunks"][chunk_id] = chunk_data
        
        return {file_id: file_data}
    
    def save_to_json(self, data: Dict[str, Any], output_path: str):
        """保存到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Parse Markdown files into structured JSON')
    parser.add_argument('input_file', help='Input Markdown file path')
    parser.add_argument('-o', '--output', help='Output JSON file path', 
                       default='markdown_output.json')
    
    args = parser.parse_args()
    
    markdown_parser = MarkdownParser()
    
    try:
        # 解析Markdown文件
        result = markdown_parser.parse_markdown_file(args.input_file)
        
        # 保存结果
        markdown_parser.save_to_json(result, args.output)
        
        print(f"Successfully parsed {args.input_file}")
        print(f"Output saved to {args.output}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 