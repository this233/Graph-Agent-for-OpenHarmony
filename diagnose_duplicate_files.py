#!/usr/bin/env python3
"""
文件重复诊断工具

用于检测文档处理过程中可能出现的重复文件内容问题
"""

import os
import hashlib
import json
from collections import defaultdict
from typing import Dict, List, Set, Any, Optional
import argparse

def compute_md5_id(content: str, prefix: str = "") -> str:
    """计算内容的MD5哈希ID"""
    return prefix + hashlib.md5(content.encode()).hexdigest()

def scan_directory_for_duplicates(directory_path: str) -> Dict[str, List[str]]:
    """
    扫描目录查找重复内容的文件
    
    Args:
        directory_path: 目录路径
        
    Returns:
        Dict: 以file_id为key，文件路径列表为value的字典
    """
    file_id_to_paths = defaultdict(list)
    total_files = 0
    
    print(f"扫描目录: {directory_path}")
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.md'):
                file_path = os.path.join(root, file)
                total_files += 1
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_id = compute_md5_id(content, "file-")
                    file_id_to_paths[file_id].append(file_path)
                    
                    if total_files % 1000 == 0:
                        print(f"  已处理 {total_files} 个文件...")
                        
                except Exception as e:
                    print(f"  错误处理文件 {file_path}: {str(e)}")
    
    print(f"扫描完成，共处理 {total_files} 个文件")
    return dict(file_id_to_paths)

def analyze_duplicates(file_id_to_paths: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    分析重复文件的统计信息
    
    Args:
        file_id_to_paths: 文件ID到路径的映射
        
    Returns:
        Dict: 分析结果
    """
    duplicates = {}
    unique_files = 0
    total_physical_files = 0
    duplicate_groups = 0
    lost_files = 0
    
    for file_id, paths in file_id_to_paths.items():
        total_physical_files += len(paths)
        
        if len(paths) > 1:
            # 发现重复文件
            duplicates[file_id] = paths
            duplicate_groups += 1
            lost_files += len(paths) - 1  # 除了第一个文件，其他都会被覆盖
        else:
            unique_files += 1
    
    return {
        'total_physical_files': total_physical_files,
        'unique_files': unique_files,
        'duplicate_groups': duplicate_groups,
        'lost_files': lost_files,
        'duplicates': duplicates
    }

def scan_json_for_duplicates(json_file_path: str) -> Dict[str, Any]:
    """
    扫描JSON文件查找重复的file、chunk、code、table的key
    
    Args:
        json_file_path: JSON文件路径
        
    Returns:
        Dict: 重复内容分析结果
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON文件不存在: {json_file_path}")
    
    print(f"扫描JSON文件: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 首先收集所有文件的信息
    file_info_map = {}  # file_id -> file_path
    
    def collect_file_info(obj, current_path=""):
        """收集文件信息"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                
                if key.startswith('file-') and isinstance(value, dict) and 'file_path' in value:
                    file_info_map[key] = value['file_path']
                
                # if isinstance(value, (dict, list)):
                #     collect_file_info(value, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
                # collect_file_info(item, new_path)
    
    # 收集文件信息
    collect_file_info(data)
    
    # 存储各种类型的key和其位置信息
    file_keys = {}  # key -> [locations]
    chunk_keys = {}
    code_keys = {}
    table_keys = {}
    jump_keys = {}
    
    def extract_file_id_from_path(path):
        """从路径中提取文件ID"""
        parts = path.split('.')
        for part in parts:
            if part.startswith('file-'):
                return part
        return None
    
    def extract_keys_recursively(obj, current_path=""):
        """递归提取各种类型的key"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                
                # # 如果当前路径包含jump，跳过其中的file_id
                # if "jump." in new_path and key.startswith('file-'):
                #     # 跳过jump字段中的file_id
                #     pass
                # else:
                # 根据key的前缀分类
                if key.startswith('file-'):
                    # if "jump." in new_path:
                    #     pass
                    # else:
                    if "jump" not in current_path:
                        if key not in file_keys:
                            file_keys[key] = []
                        file_keys[key].append({
                            'location': new_path,
                            'content': value.get('content', '') if isinstance(value, dict) else str(value),
                            'file_path': value.get('file_path', '') if isinstance(value, dict) else ''
                        })
                elif key.startswith('chunk-'):
                    if key not in chunk_keys:
                        chunk_keys[key] = []
                    
                    # 从路径中提取文件ID，获取对应的文件路径
                    file_id = extract_file_id_from_path(new_path)
                    file_path = file_info_map.get(file_id, 'Unknown')
                    
                    chunk_keys[key].append({
                        'location': new_path,
                        'content': value.get('content', '') if isinstance(value, dict) else str(value),
                        'file_id': file_id,
                        'file_path': file_path
                    })
                elif key.startswith('code-'):
                    if key not in code_keys:
                        code_keys[key] = []
                    
                    # 从路径中提取文件ID，获取对应的文件路径
                    file_id = extract_file_id_from_path(new_path)
                    file_path = file_info_map.get(file_id, 'Unknown')
                    
                    code_keys[key].append({
                        'location': new_path,
                        'content': value.get('content', '') if isinstance(value, dict) else str(value),
                        'abstract': value.get('abstract', '') if isinstance(value, dict) else '',
                        'file_id': file_id,
                        'file_path': file_path
                    })
                elif key.startswith('table-'):
                    if key not in table_keys:
                        table_keys[key] = []
                    
                    # 从路径中提取文件ID，获取对应的文件路径
                    file_id = extract_file_id_from_path(new_path)
                    file_path = file_info_map.get(file_id, 'Unknown')
                    
                    table_keys[key].append({
                        'location': new_path,
                        'content': value.get('content', '') if isinstance(value, dict) else str(value),
                        'abstract': value.get('abstract', '') if isinstance(value, dict) else '',
                        'file_id': file_id,
                        'file_path': file_path
                    })
                # elif key.startswith('jump-'):
                #     if key not in jump_keys:
                #         jump_keys[key] = []
                #     jump_keys[key].append({
                #         'location': new_path,
                #         'content': str(value)
                #     })
                
                # 递归处理值
                if isinstance(value, (dict, list)):
                    extract_keys_recursively(value, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
                extract_keys_recursively(item, new_path)
    
    # 提取所有key
    extract_keys_recursively(data)
    
    # 找出重复的key
    def find_duplicate_keys(keys_dict):
        """找出重复的key"""
        duplicates = {}
        for key, locations in keys_dict.items():
            if len(locations) > 1:
                duplicates[key] = locations
        return duplicates
    
    file_duplicates = find_duplicate_keys(file_keys)
    chunk_duplicates = find_duplicate_keys(chunk_keys)
    code_duplicates = find_duplicate_keys(code_keys)
    table_duplicates = find_duplicate_keys(table_keys)
    jump_duplicates = find_duplicate_keys(jump_keys)
    
    print(f"JSON扫描完成:")
    print(f"  文件总数: {len(file_keys)}, 重复: {len(file_duplicates)}")
    print(f"  块总数: {len(chunk_keys)}, 重复: {len(chunk_duplicates)}")
    print(f"  代码块总数: {len(code_keys)}, 重复: {len(code_duplicates)}")
    print(f"  表格总数: {len(table_keys)}, 重复: {len(table_duplicates)}")
    print(f"  跳转总数: {len(jump_keys)}, 重复: {len(jump_duplicates)}")
    
    return {
        'file_duplicates': file_duplicates,
        'chunk_duplicates': chunk_duplicates,
        'code_duplicates': code_duplicates,
        'table_duplicates': table_duplicates,
        'jump_duplicates': jump_duplicates,
        'stats': {
            'total_files': len(file_keys),
            'total_chunks': len(chunk_keys),
            'total_codes': len(code_keys),
            'total_tables': len(table_keys),
            'total_jumps': len(jump_keys),
            'duplicate_files': len(file_duplicates),
            'duplicate_chunks': len(chunk_duplicates),
            'duplicate_codes': len(code_duplicates),
            'duplicate_tables': len(table_duplicates),
            'duplicate_jumps': len(jump_duplicates)
        }
    }

def print_analysis_results(analysis: Dict[str, Any]):
    """打印分析结果"""
    print("\n=== 文件重复分析结果 ===")
    print(f"物理文件总数: {analysis['total_physical_files']}")
    print(f"唯一文件数: {analysis['unique_files']}")
    print(f"重复文件组数: {analysis['duplicate_groups']}")
    print(f"被覆盖的文件数: {analysis['lost_files']}")
    print(f"最终保留的文件数: {analysis['unique_files']}")
    
    if analysis['duplicate_groups'] > 0:
        print(f"\n预期差异: {analysis['total_physical_files']} - {analysis['unique_files']} = {analysis['lost_files']}")
        
        print("\n=== 重复文件详情 ===")
        for i, (file_id, paths) in enumerate(analysis['duplicates'].items(), 1):
            print(f"\n重复组 {i} (file_id: {file_id}):")
            for j, path in enumerate(paths, 1):
                status = "保留" if j == len(paths) else "覆盖"
                print(f"  [{status}] {path}")

def print_json_analysis_results(analysis: Dict[str, Any]):
    """打印JSON分析结果"""
    stats = analysis['stats']
    
    print("\n=== JSON内容重复分析结果 ===")
    print(f"文件总数: {stats['total_files']}, 重复key组数: {stats['duplicate_files']}")
    print(f"块总数: {stats['total_chunks']}, 重复key组数: {stats['duplicate_chunks']}")
    print(f"代码块总数: {stats['total_codes']}, 重复key组数: {stats['duplicate_codes']}")
    print(f"表格总数: {stats['total_tables']}, 重复key组数: {stats['duplicate_tables']}")
    print(f"跳转总数: {stats['total_jumps']}, 重复key组数: {stats['duplicate_jumps']}")
    
    # 显示重复文件详情
    if analysis['file_duplicates']:
        print(f"\n=== 重复文件详情 ===")
        for i, (key, locations) in enumerate(analysis['file_duplicates'].items(), 1):
            print(f"\n重复组 {i} (key: {key}):")
            print(f"  出现次数: {len(locations)}")
            for j, location_info in enumerate(locations, 1):
                print(f"  [{j}] 位置: {location_info['location']}")
                if location_info['file_path']:
                    print(f"      文件路径: {location_info['file_path']}")
                if location_info['content']:
                    preview = location_info['content'][:100] + "..." if len(location_info['content']) > 100 else location_info['content']
                    print(f"      内容预览: {preview}")
    
    # 显示重复块详情
    if analysis['chunk_duplicates']:
        print(f"\n=== 重复块详情 ===")
        for i, (key, locations) in enumerate(analysis['chunk_duplicates'].items(), 1):
            print(f"\n重复组 {i} (key: {key}):")
            print(f"  出现次数: {len(locations)}")
            for j, location_info in enumerate(locations, 1):
                print(f"  [{j}] 位置: {location_info['location']}")
                if location_info.get('file_path'):
                    print(f"      所属文件: {location_info['file_path']}")
                if location_info.get('file_id'):
                    print(f"      文件ID: {location_info['file_id']}")
                if location_info['content']:
                    preview = location_info['content'][:100] + "..." if len(location_info['content']) > 100 else location_info['content']
                    print(f"      内容预览: {preview}")
    
    # 显示重复代码块详情
    if analysis['code_duplicates']:
        print(f"\n=== 重复代码块详情 ===")
        for i, (key, locations) in enumerate(analysis['code_duplicates'].items(), 1):
            print(f"\n重复组 {i} (key: {key}):")
            print(f"  出现次数: {len(locations)}")
            for j, location_info in enumerate(locations, 1):
                print(f"  [{j}] 位置: {location_info['location']}")
                if location_info.get('file_path'):
                    print(f"      所属文件: {location_info['file_path']}")
                if location_info.get('file_id'):
                    print(f"      文件ID: {location_info['file_id']}")
                if location_info['content']:
                    preview = location_info['content'][:100] + "..." if len(location_info['content']) > 100 else location_info['content']
                    print(f"      内容预览: {preview}")
    
    # 显示重复表格详情
    if analysis['table_duplicates']:
        print(f"\n=== 重复表格详情 ===")
        for i, (key, locations) in enumerate(analysis['table_duplicates'].items(), 1):
            print(f"\n重复组 {i} (key: {key}):")
            print(f"  出现次数: {len(locations)}")
            for j, location_info in enumerate(locations, 1):
                print(f"  [{j}] 位置: {location_info['location']}")
                if location_info.get('file_path'):
                    print(f"      所属文件: {location_info['file_path']}")
                if location_info.get('file_id'):
                    print(f"      文件ID: {location_info['file_id']}")
                if location_info['content']:
                    preview = location_info['content'][:100] + "..." if len(location_info['content']) > 100 else location_info['content']
                    print(f"      内容预览: {preview}")
    
    # 显示重复跳转详情
    if analysis['jump_duplicates']:
        print(f"\n=== 重复跳转详情 ===")
        for i, (key, locations) in enumerate(analysis['jump_duplicates'].items(), 1):
            print(f"\n重复组 {i} (key: {key}):")
            print(f"  出现次数: {len(locations)}")
            for j, location_info in enumerate(locations, 1):
                print(f"  [{j}] 位置: {location_info['location']}")
                if location_info['content']:
                    print(f"      内容: {location_info['content']}")

def save_duplicate_report(analysis: Dict[str, Any], output_file: str):
    """保存重复文件报告"""
    report = {
        'summary': {
            'total_physical_files': analysis['total_physical_files'],
            'unique_files': analysis['unique_files'],
            'duplicate_groups': analysis['duplicate_groups'],
            'lost_files': analysis['lost_files']
        },
        'duplicate_details': []
    }
    
    for file_id, paths in analysis['duplicates'].items():
        detail = {
            'file_id': file_id,
            'file_count': len(paths),
            'files': []
        }
        
        for i, path in enumerate(paths):
            detail['files'].append({
                'path': path,
                'status': 'retained' if i == len(paths) - 1 else 'overwritten'
            })
        
        report['duplicate_details'].append(detail)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n重复文件报告已保存到: {output_file}")

def save_json_duplicate_report(analysis: Dict[str, Any], output_file: str):
    """保存JSON重复内容报告"""
    report = {
        'summary': analysis['stats'],
        'duplicate_details': {
            'files': [],
            'chunks': [],
            'codes': [],
            'tables': [],
            'jumps': []
        }
    }
    
    # 保存文件重复详情
    for key, locations in analysis['file_duplicates'].items():
        detail = {
            'key': key,
            'count': len(locations),
            'locations': []
        }
        for location_info in locations:
            detail['locations'].append({
                'location': location_info['location'],
                'file_path': location_info['file_path'],
                'content_preview': location_info['content'][:200] + "..." if len(location_info['content']) > 200 else location_info['content']
            })
        report['duplicate_details']['files'].append(detail)
    
    # 保存块重复详情
    for key, locations in analysis['chunk_duplicates'].items():
        detail = {
            'key': key,
            'count': len(locations),
            'locations': []
        }
        for location_info in locations:
            location_detail = {
                'location': location_info['location'],
                'content_preview': location_info['content'][:200] + "..." if len(location_info['content']) > 200 else location_info['content']
            }
            if location_info.get('file_path'):
                location_detail['file_path'] = location_info['file_path']
            if location_info.get('file_id'):
                location_detail['file_id'] = location_info['file_id']
            detail['locations'].append(location_detail)
        report['duplicate_details']['chunks'].append(detail)
    
    # 保存代码块重复详情
    for key, locations in analysis['code_duplicates'].items():
        detail = {
            'key': key,
            'count': len(locations),
            'locations': []
        }
        for location_info in locations:
            location_detail = {
                'location': location_info['location'],
                'abstract': location_info['abstract'],
                'content_preview': location_info['content'][:200] + "..." if len(location_info['content']) > 200 else location_info['content']
            }
            if location_info.get('file_path'):
                location_detail['file_path'] = location_info['file_path']
            if location_info.get('file_id'):
                location_detail['file_id'] = location_info['file_id']
            detail['locations'].append(location_detail)
        report['duplicate_details']['codes'].append(detail)
    
    # 保存表格重复详情
    for key, locations in analysis['table_duplicates'].items():
        detail = {
            'key': key,
            'count': len(locations),
            'locations': []
        }
        for location_info in locations:
            location_detail = {
                'location': location_info['location'],
                'abstract': location_info['abstract'],
                'content_preview': location_info['content'][:200] + "..." if len(location_info['content']) > 200 else location_info['content']
            }
            if location_info.get('file_path'):
                location_detail['file_path'] = location_info['file_path']
            if location_info.get('file_id'):
                location_detail['file_id'] = location_info['file_id']
            detail['locations'].append(location_detail)
        report['duplicate_details']['tables'].append(detail)
    
    # 保存跳转重复详情
    for key, locations in analysis['jump_duplicates'].items():
        detail = {
            'key': key,
            'count': len(locations),
            'locations': []
        }
        for location_info in locations:
            detail['locations'].append({
                'location': location_info['location'],
                'content': location_info['content']
            })
        report['duplicate_details']['jumps'].append(detail)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nJSON重复内容报告已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="诊断文档处理中的重复文件问题")
    parser.add_argument("input_path", help="要扫描的目录路径或JSON文件路径")
    parser.add_argument("--output", "-o", help="输出报告文件路径")
    parser.add_argument("--show-details", "-d", action="store_true", help="显示详细的重复文件信息")
    parser.add_argument("--json-mode", "-j", action="store_true", help="分析JSON文件中的重复key")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"错误: 路径不存在: {args.input_path}")
        return
    
    if args.json_mode:
        # JSON模式：分析JSON文件中的重复key
        try:
            analysis = scan_json_for_duplicates(args.input_path)
            print_json_analysis_results(analysis)
            
            if args.output:
                save_json_duplicate_report(analysis, args.output)
        except Exception as e:
            print(f"JSON分析出错: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        # 目录模式：扫描目录中的重复文件
        if not os.path.isdir(args.input_path):
            print(f"错误: 非JSON模式需要提供目录路径")
            return
        
        # 扫描目录
        file_id_to_paths = scan_directory_for_duplicates(args.input_path)
        
        # 分析结果
        analysis = analyze_duplicates(file_id_to_paths)
        
        # 打印结果
        print_analysis_results(analysis)
        
        if args.show_details and analysis['duplicate_groups'] > 0:
            print("\n=== 详细重复文件列表 ===")
            for file_id, paths in analysis['duplicates'].items():
                print(f"\nFile ID: {file_id}")
                for path in paths:
                    print(f"  - {path}")
        
        # 保存报告
        if args.output:
            save_duplicate_report(analysis, args.output)

if __name__ == "__main__":
    main() 