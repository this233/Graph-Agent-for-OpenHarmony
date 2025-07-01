import json
import hashlib
import os
import re
from typing import List, Dict, Any, Tuple, Optional
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self, context_lines: int = 5):
        """
        初始化文档处理器
        
        Args:
            context_lines: 上下文行数
        """
        self.context_lines = context_lines
        
        # 定义 Markdown 标题层级
        self.headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"), 
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
        )

    def _compute_md5_id(self, content: str, prefix: str = "") -> str:
        """计算内容的MD5哈希ID"""
        return prefix + hashlib.md5(content.encode()).hexdigest()

    def _extract_hyperlinks(self, content: str, current_file_path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """
        提取内容中的超链接
        
        Args:
            content: 文本内容
            current_file_path: 当前文件路径，用于解析相对路径
            
        Returns:
            Dict: 超链接信息字典
        """
        jump_dict = {}
        # 匹配 [text](link) 格式的链接
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(pattern, content)
        
        for text, link in matches:
            # 判断是否是路径名（不是URL）
            if not (link.startswith('http://') or link.startswith('https://') or link.startswith('ftp://')):
                # 判断是否是md文件
                if link.lower().endswith('.md'):
                    # 处理相对路径
                    if current_file_path and not os.path.isabs(link):
                        # 相对路径：基于当前文件所在目录解析
                        current_dir = os.path.dirname(current_file_path)
                        absolute_path = os.path.normpath(os.path.join(current_dir, link))
                    else:
                        # 绝对路径或没有当前文件路径信息
                        absolute_path = link
                    
                    # 尝试读取文件内容
                    try:
                        if os.path.exists(absolute_path):
                            with open(absolute_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                            # 基于文件内容计算md5
                            file_id = self._compute_md5_id(file_content, "file-")
                            jump_dict[file_id] = {
                                "file_path": link,
                                "jump_name": text
                            }
                    except Exception:
                        # 读取失败，不添加到jump_dict中
                        pass
            # else:
            #     # 是URL，使用链接本身计算md5
            #     file_id = self._compute_md5_id(link, "file-")
            #     jump_dict[file_id] = {
            #         "file_path": link
            #     }
        
        return jump_dict
    def _extract_code_blocks(self, content: str) -> Tuple[Dict[str, Dict[str, str]], str]:
        """
        提取代码块并返回过滤后的内容
        
        Args:
            content: 文本内容
            
        Returns:
            Tuple: (代码块字典, 移除长代码块后的内容)
        """
        codes_dict = {}
        lines = content.split('\n')
        filtered_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 检测代码块开始
            if line.startswith('```'):
                start_line = i
                code_content_lines = [lines[i]]  # 包含开始的```行
                i += 1
                
                # 寻找结束的```
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_content_lines.append(lines[i])
                    i += 1
                
                if i < len(lines):  # 找到了结束标记
                    code_content_lines.append(lines[i])  # 包含结束的```行
                    end_line = i
                    
                    # 检查代码块长度（不包括开始和结束的```行）
                    code_lines_count = len(code_content_lines) - 2  # 减去开始和结束的```行
                    
                    # 只有长度>=5行的代码块才提取到codes_dict并从原文中移除
                    if code_lines_count >= 5:
                        # 提取带上下文的代码块
                        context_start = max(0, start_line - self.context_lines)
                        context_end = min(len(lines) - 1, end_line + self.context_lines)
                        
                        # 构建带上下文的代码块内容
                        context_lines_list = []
                        
                        # 添加前置上下文
                        if context_start < start_line:
                            context_lines_list.extend(lines[context_start:start_line])
                        
                        # 添加代码块本身
                        context_lines_list.extend(code_content_lines)
                        
                        # 添加后置上下文
                        if end_line + 1 <= context_end:
                            context_lines_list.extend(lines[end_line + 1:context_end + 1])
                        
                        code_with_context = '\n'.join(context_lines_list)
                        code_id = self._compute_md5_id(code_with_context, "code-")
                        
                        codes_dict[code_id] = {
                            "abstract": "",
                            "content": code_with_context
                        }
                        
                        # # 长代码块从过滤后的内容中移除（用空行替换）
                        # for _ in range(start_line, end_line + 1):
                        #     filtered_lines.append("")
                    else:
                        # 短代码块（<5行）保留在原文中
                        filtered_lines.extend(code_content_lines)
                else:
                    # 如果没找到结束标记，保留原内容
                    filtered_lines.extend(code_content_lines)
                    
                i += 1
            else:
                filtered_lines.append(lines[i])
                i += 1
        
        filtered_content = '\n'.join(filtered_lines)
        return codes_dict, filtered_content

    def _extract_tables(self, content: str) -> Tuple[Dict[str, Dict[str, str]], str]:
        """
        提取表格并返回过滤后的内容
        
        Args:
            content: 文本内容
            
        Returns:
            Tuple: (表格字典, 移除长表格后的内容)
        """
        tables_dict = {}
        lines = content.split('\n')
        filtered_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 检测HTML表格
            if line.lower().startswith('<table'):
                start_line = i
                table_lines = [lines[i]]
                i += 1
                
                # 查找</table>结束
                while i < len(lines) and '</table>' not in lines[i].lower():
                    table_lines.append(lines[i])
                    i += 1
                
                if i < len(lines):
                    table_lines.append(lines[i])  # 包含</table>行
                    end_line = i
                    
                    # 检查HTML表格长度，只有>=5行才提取到tables_dict并从原文中移除
                    if len(table_lines) >= 5:
                        # 提取带上下文的表格
                        context_start = max(0, start_line - self.context_lines)
                        context_end = min(len(lines) - 1, end_line + self.context_lines)
                        
                        context_lines_list = []
                        if context_start < start_line:
                            context_lines_list.extend(lines[context_start:start_line])
                        context_lines_list.extend(table_lines)
                        if end_line + 1 <= context_end:
                            context_lines_list.extend(lines[end_line + 1:context_end + 1])
                        
                        table_with_context = '\n'.join(context_lines_list)
                        table_id = self._compute_md5_id(table_with_context, "table-")
                        
                        tables_dict[table_id] = {
                            "abstract": "",
                            "content": table_with_context
                        }
                        
                        # # 长表格从过滤后的内容中移除（用空行替换）
                        # for _ in range(start_line, end_line + 1):
                        #     filtered_lines.append("")
                    else:
                        # 短表格（<5行）保留在原文中
                        filtered_lines.extend(table_lines)
                else:
                    # 如果没找到结束标记，保留原内容
                    filtered_lines.extend(table_lines)
                    
                i += 1
                
            # 检测Markdown表格（|分隔的表格）
            elif '|' in line:
                start_line = i
                table_lines = []
                
                # 收集连续的包含|的行
                while i < len(lines) and '|' in lines[i]:
                    table_lines.append(lines[i])
                    i += 1
                
                # 只有当表格行数>=5时才提取到tables_dict并从原文中移除
                if len(table_lines) >= 5:
                    end_line = start_line + len(table_lines) - 1
                    
                    # 提取带上下文的表格
                    context_start = max(0, start_line - self.context_lines)
                    context_end = min(len(lines) - 1, end_line + self.context_lines)
                    
                    context_lines_list = []
                    if context_start < start_line:
                        context_lines_list.extend(lines[context_start:start_line])
                    context_lines_list.extend(table_lines)
                    if end_line + 1 <= context_end:
                        context_lines_list.extend(lines[end_line + 1:context_end + 1])
                    
                    table_with_context = '\n'.join(context_lines_list)
                    table_id = self._compute_md5_id(table_with_context, "table-")
                    
                    tables_dict[table_id] = {
                        "abstract": "",
                        "content": table_with_context
                    }
                    
                    # # 长表格从过滤后的内容中移除（用空行替换）
                    # for _ in range(len(table_lines)):
                    #     filtered_lines.append("")
                else:
                    # 短表格（<5行）保留在原文中
                    filtered_lines.extend(table_lines)
            else:
                filtered_lines.append(lines[i])
                i += 1
        
        filtered_content = '\n'.join(filtered_lines)
        return tables_dict, filtered_content

    def _process_chunk_content(self, content: str, current_file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        处理单个chunk的内容，提取超链接、代码块、表格等
        
        Args:
            content: chunk内容
            
        Returns:
            Dict: 处理后的chunk信息
        """
        # 提取超链接
        jump_dict = self._extract_hyperlinks(content, current_file_path)
        
        # 提取代码块
        codes_dict, content_without_codes = self._extract_code_blocks(content)
        
        # 从移除代码块后的内容中提取表格
        tables_dict, filtered_content = self._extract_tables(content_without_codes)
        
        # 创建chunk信息
        chunk_info = {
            "abstract": "",
            "content": content,
            "jump": jump_dict,
            "codes": codes_dict,
            "tables": tables_dict,
            "filter_chunk": {
                "content": filtered_content.strip(),
                "extracted_entities": [],
                "extracted_triples": []
            }
        }
        
        return chunk_info

    def _recursive_chunk_split(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        对内容进行分块，先用markdown_splitter分块，然后按metadata长度排序处理
        
        Args:
            content: 要分块的内容
            
        Returns:
            Dict: 分块结果
        """
        chunks_dict = {}
        
        try:
            # 使用self.markdown_splitter一次性分块
            splits = self.markdown_splitter.split_text(content)
            
            if not splits:
                return chunks_dict
            
            # 按照len(metadata.keys())排序，从短到长处理
            # splits_with_metadata = [split for split in splits if split.metadata]
            for split in splits:
                if not split.metadata:
                    split.metadata = {}
            splits.sort(key=lambda x: len(x.metadata.keys()))
            
            # 存储所有chunk信息，用于建立父子关系
            all_chunks = {}  # chunk_id -> (chunk_info, metadata, split)
            
            # 处理有metadata的splits
            for split in splits:
                # 构建chunk内容，先添加metadata中的标题
                chunk_content = ""
                if split.metadata:
                    # 按标题层级顺序添加标题
                    for header_level in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        if header_level in split.metadata:
                            header_prefix = "#" * int(header_level[1:])
                            chunk_content += f"{header_prefix} {split.metadata[header_level]}  \n"
                
                # 添加页面内容
                chunk_content += split.page_content
                
                if not chunk_content.strip():
                    continue
                # 生成chunk ID
                chunk_id = self._compute_md5_id(chunk_content, "chunk-")
                
                # 处理chunk内容
                chunk_info = self._process_chunk_content(chunk_content, file_path)
                chunk_info["metadata"] = split.metadata
                
                # 存储chunk信息
                all_chunks[chunk_id] = (chunk_info, split.metadata, split)
            
            
            # 建立父子关系
            for chunk_id, (chunk_info, metadata, split) in all_chunks.items():
                parent_chunk_id = self._find_parent_chunk(chunk_id, metadata, all_chunks)
                
                if parent_chunk_id:
                    # 有父chunk，放到父chunk的chunks字段里
                    parent_chunk_info = all_chunks[parent_chunk_id][0]
                    if "chunks" not in parent_chunk_info:
                        parent_chunk_info["chunks"] = {}
                    parent_chunk_info["chunks"][chunk_id] = chunk_info
                else:
                    # 没有父chunk，放到顶级chunks字典里
                    chunks_dict[chunk_id] = chunk_info
                
        except Exception as e:
            # print(f"Warning: Error in markdown splitting: {str(e)}")
            # 如果分割失败，不创建chunk
            pass
        
        return chunks_dict
    
    def _find_parent_chunk(self, current_chunk_id: str, current_metadata: Dict, all_chunks: Dict) -> Optional[str]:
        """
        找到当前chunk的父chunk
        
        Args:
            current_chunk_id: 当前chunk的ID
            current_metadata: 当前chunk的metadata
            all_chunks: 所有chunk的信息
            
        Returns:
            父chunk的ID，如果没有找到则返回None
        """
        if len(current_metadata) <= 1:
            # 顶级chunk没有父chunk
            return None
        
        # 寻找父chunk：metadata长度比当前chunk少1，且是当前metadata的子集
        for target_length in range(len(current_metadata) - 1, 0, -1):
            for chunk_id, (chunk_info, metadata, split) in all_chunks.items():
                if chunk_id == current_chunk_id:
                    continue
                
                if len(metadata) == target_length:
                    is_parent = True
                    for key, value in metadata.items():
                        if key not in current_metadata or current_metadata[key] != value:
                            is_parent = False
                            break
                    
                    if is_parent:
                        return chunk_id
        
        return None
    def _add_heading_numbers(self, content: str) -> str:
        """
        为标题添加层级序号（如果标题已有序号则覆盖）
        
        Args:
            content: 原始内容
            
        Returns:
            str: 添加序号后的内容
        """
        lines = content.split('\n')
        result_lines = []
        
        # 用于跟踪各级标题的计数器
        counters = [0, 0, 0, 0, 0, 0]  # 对应h1-h6
        
        in_code_block = False
        
        for line in lines:
            stripped_line = line.strip()
            
            # 检查是否进入或退出代码块
            if stripped_line.startswith('```'):
                in_code_block = not in_code_block
                result_lines.append(line)
                continue
            
            # 如果在代码块内，直接添加行，不处理标题
            if in_code_block:
                result_lines.append(line)
                continue
            
            # 检查是否是标题行（必须以#开头，且#后面必须有空格或直接是内容）
            if stripped_line.startswith('#'):
                # 计算标题级别
                level = 0
                for char in stripped_line:
                    if char == '#':
                        level += 1
                    else:
                        break
                
                # 检查是否是有效的标题格式
                # 标题必须满足：1-6个#，且#后面要么是空格，要么直接是内容
                if 1 <= level <= 6 and len(stripped_line) > level:
                    # 检查#后面的字符
                    char_after_hash = stripped_line[level] if level < len(stripped_line) else ''
                    
                    # 如果#后面是空格或者直接是内容（非空格字符），则认为是标题
                    if char_after_hash == ' ' or (char_after_hash != '' and not char_after_hash.isspace()):
                        # 更新计数器
                        counters[level - 1] += 1
                        # 重置更深层级的计数器
                        for i in range(level, 6):
                            counters[i] = 0
                        
                        # 提取标题文本（去掉#和空格）
                        title_text = stripped_line[level:].strip()
                        
                        # 移除标题文本中已有的序号（匹配各种序号格式）
                        # 匹配模式：数字.数字.数字... 或 数字、 或 数字) 等
                        import re
                        # 移除开头的序号模式
                        title_text = re.sub(r'^[\d\.\-\s\)、）]+', '', title_text).strip()
                        
                        # 构建序号
                        number_parts = []
                        for i in range(level):
                            if counters[i] > 0:
                                number_parts.append(str(counters[i]))
                        
                        if number_parts:
                            number_str = '.'.join(number_parts)
                            # 构建新的标题行
                            new_title = '#' * level + ' ' + number_str + ' ' + title_text
                            result_lines.append(new_title)
                        else:
                            result_lines.append(line)
                    else:
                        # 不是有效的标题格式（可能是注释等），直接添加
                        result_lines.append(line)
                else:
                    # 不是有效的标题格式，直接添加
                    result_lines.append(line)
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines)

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        处理单个文件并生成JSON结构
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict: 文件处理结果
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not file_path.lower().endswith('.md'):
            raise ValueError(f"Only markdown files are supported, got: {file_path}")
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 生成文件ID
        file_id = self._compute_md5_id(content, "file-")
        
        # 为标题添加层级序号
        content_with_numbers = self._add_heading_numbers(content)
        
        # 进行递归分块
        chunks_dict = self._recursive_chunk_split(content_with_numbers, file_path)
        
        # 构建文件信息
        file_info = {
            file_id: {
                "abstract": "",
                "file_path": file_path,
                "content": content_with_numbers,
                "chunks": chunks_dict
            }
        }
        
        return file_info
    def process_directory(self, directory_path: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        处理目录下的所有Markdown文件
        
        Args:
            directory_path: 目录路径
            output_file: 输出JSON文件路径（可选）
            
        Returns:
            Dict: 所有文件的处理结果
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_files_info = {}
        
        # 遍历目录下的所有文件
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.md'):
                    file_path = os.path.join(root, file)
                    try:
                        file_info = self.process_file(file_path)
                        all_files_info.update(file_info)
                        print(f"Processed: {file_path}")
                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")
                        continue
        
        # 保存到JSON文件
        if output_file:
            self.save_to_json(all_files_info, output_file)
            
        # 递归统计chunks、tables、codes字段数量
        chunks_count, tables_count, codes_count, jump_count = self._count_fields_recursively(all_files_info)
        print(f"Statistics:")
        print(f"  Total chunks: {chunks_count}")
        print(f"  Total tables: {tables_count}")
        print(f"  Total codes: {codes_count}")
        print(f"  Total jump: {jump_count}")
        
        
        
        
        return all_files_info
    
    def _count_fields_recursively(self, data: Dict[str, Any]) -> tuple:
        """
        递归统计chunks、tables、codes字段数量
        
        Args:
            data: 要统计的数据字典
            
        Returns:
            tuple: (chunks_count, tables_count, codes_count)
        """
        chunks_count = 0
        tables_count = 0
        codes_count = 0
        jump_count = 0
        
        def _recursive_count(obj):
            nonlocal chunks_count, tables_count, codes_count, jump_count
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "chunks" and isinstance(value, dict):
                        chunks_count += len(value)
                    elif key == "tables" and isinstance(value, dict):
                        tables_count += len(value)
                    elif key == "codes" and isinstance(value, dict):
                        codes_count += len(value)
                    elif key == "jump" and isinstance(value, dict):
                        jump_count += len(value)
                    # 递归处理嵌套的字典和列表
                    _recursive_count(value)
            elif isinstance(obj, list):
                for item in obj:
                    _recursive_count(item)
        
        _recursive_count(data)
        return chunks_count, tables_count, codes_count, jump_count

    def save_to_json(self, data: Dict[str, Any], output_file: str):
        """
        保存数据到JSON文件
        
        Args:
            data: 要保存的数据
            output_file: 输出文件路径
        """
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving to JSON: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 创建处理器实例
    processor = DocumentProcessor(context_lines=15)
    
    # 处理单个文件
    # /root/code/docs/zh-cn/application-dev/Readme-CN.md 50 102
    # /root/code/docs/zh-cn/glossary.md 72 133
    # /root/code/docs/zh-cn/OpenHarmony-Overview_zh.md 95 193
    # /root/code/docs/zh-cn/device-dev/kernel/kernel-mini-basic-time.md 35 77
    # /root/code/docs/zh-cn/application-dev/reference/apis-arkui/_ark_u_i___native_module.md 4914 7903
    # /root/code/docs/zh-cn/release-notes/api-diff/v4.0-Release to v3.2-Release/js-apidiff-arkui.md 0 0
    # /root/code/docs/zh-cn/application-dev/reference/apis-arkui/js-apis-window.md 2834 4053

    # try:
    #     result = processor.process_file("/root/code/docs/zh-cn/OpenHarmony-Overview_zh.md")
    #     processor.save_to_json(result, "outputs/Harmony_docs_zh_cn/markdown_parse/structure.json")
    #     print("Single file processing completed!")
    # except Exception as e:
    #     print(f"Single file processing failed: {str(e)}")
    
    # 处理整个目录 /root/code/docs/zh-cn
    try:
        result = processor.process_directory("/root/code/docs/zh-cn", "outputs/Harmony_docs_zh_cn/markdown_parse/structure.json")
        print("Directory processing completed!")
    except Exception as e:
        print(f"Directory processing failed: {str(e)}")
