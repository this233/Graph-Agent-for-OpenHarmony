import os
from typing import List
import json
import argparse
import logging

from src.hipporag import HippoRAG

# 清除代理设置
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

def load_triples_json(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

os.environ['OPENAI_API_KEY'] = 'BQm_Gkd1EoTcHkJfVf31dTWfMIOsW3_mKIDfM5j-MvvwNM5jNl9XnLOjvNjEOuDiIWoKb-DIphdRWt2gOoNwBw'

def main():

    # Prepare datasets and evaluation
    # docs = [
    #     "Oliver Badman is a politician.",
    #     "George Rankin is a politician.",
    #     "Thomas Marwick is a politician.",
    #     "Cinderella attended the royal ball.",
    #     "The prince used the lost glass slipper to search the kingdom.",
    #     "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    #     "Erik Hort's birthplace is Montebello.",
    #     "Marina is bom in Minsk.",
    #     "Montebello is a part of Rockland County."
    # ]
    my_json_path = "outputs/Harmony_docs_zh_cn/markdown_parse/triples.json"
    my_json_data = load_triples_json(my_json_path)

    # save_dir = 'outputs/openai'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    # llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name
    # embedding_model_name = 'text-embedding-3-small'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    save_dir = 'outputs/Harmony_docs_zh_cn'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    # llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name
    llm_model_name = 'deepseek-v3.2-exp'  # 华为 ModelArts MaaS 模型名称
    # embedding_model_name = 'text-embedding-3-small'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    embedding_model_name = 'Qwen3-Embedding-4B'

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url='https://api.modelarts-maas.com/openai/v1',  # OpenAI 兼容接口
                        # embedding_base_url='https://api.vveai.com/v1',
                        embedding_model_name=embedding_model_name
                        # embedding_base_url='127.0.0.1:11434',
                        # embedding_model_name='dengcao/Qwen3-Embedding-8B:Q5_K_M'
                        )

    # ====== 查询列表 ======
    queries = [
        "What is OpenHarmony's purpose?",
        "What is OpenHarmony's architecture?",
        "What is OpenHarmony's core features?",
        "What is OpenHarmony's development model?",
        "What is OpenHarmony's development process?",
        "What is OpenHarmony's development tools?",
        "What is OpenHarmony's development environment?",
        "What is OpenHarmony's development language?",
        "What is OpenHarmony's development framework?",
    ]

    # ====== 选择运行模式 ======
    # 1. 重新建图（首次运行或需要更新映射时需要）
    # 2. 使用新的retrieve_v2检索
    
    RUN_INDEXING = False  # 设为True重新建图
    
    if RUN_INDEXING:
        print("🔧 开始建图...")
        hipporag.index_from_json(my_json_data)
        print("✅ 建图完成！")
        return
    
    # ====== 使用新的 retrieve_v2 检索流程 ======
    # 流程:
    # 1. Embedding候选获取: Fact(100), File(50), Chunk(50)
    # 2. LLM Rerank: Fact(10), File(5), Chunk(5)  
    # 3. 图扩散: 必选Chunk + 扩散候选(考虑边代价+Query相似度)
    # 4. 最终LLM Rerank: Chunk(10), Code(3), Table(3), Image(3)
    
    print(f"\n📋 retrieve_v2 检索配置:")
    print(f"   阶段1: Fact候选100个, File候选50个, Chunk候选50个")
    print(f"   阶段2: Fact保留10个, File保留5个, Chunk保留5个")
    print(f"   阶段3: 图扩散(Chunk15, Code10, Table10, Image10)")
    print(f"   阶段4: 最终Chunk10个, Code/Table/Image各3个")
    print(f"   阶段5: LLM报告生成（整合信息供前端页面生成器使用）")
    
    results = hipporag.retrieve_v2(
        queries=queries,  # 先只测试前2个查询
        # 阶段1参数
        fact_candidate_k=50,
        file_candidate_k=30,
        chunk_candidate_k=30,
        # 阶段2参数
        fact_top_k=10,
        file_top_k=10,
        chunk_top_k=10,
        # 阶段3参数
        spread_chunk_k=30,
        spread_code_k=10,
        spread_table_k=10,
        spread_image_k=10,
        # 阶段4参数
        final_chunk_k=10,
        final_code_k=3,
        final_table_k=3,
        final_image_k=3,
        # 阶段5参数
        generate_report=True,  # 开启LLM报告生成
        verbose=True
    )
    
    # ====== 最终汇总 ======
    print(f"\n{'#'*70}")
    print(f"# 最终检索结果汇总 (retrieve_v2)")
    print(f"{'#'*70}")
    
    for i, r in enumerate(results):
        print(f"\n查询 {i+1}: {r['query']}")
        print(f"  📄 Chunk: {len(r['chunks']['ids'])}个")
        for j, chunk_id in enumerate(r['chunks']['ids'][:3]):
            content_preview = r['chunks']['contents'][j][:100] if j < len(r['chunks']['contents']) else ''
            print(f"      [{j+1}] {chunk_id}")
            print(f"          {content_preview}...")
        
        print(f"  💻 Code: {len(r['codes']['ids'])}个")
        for j, code_id in enumerate(r['codes']['ids'][:2]):
            print(f"      [{j+1}] {code_id}")
        
        print(f"  📊 Table: {len(r['tables']['ids'])}个")
        for j, table_id in enumerate(r['tables']['ids'][:2]):
            print(f"      [{j+1}] {table_id}")
        
        print(f"  🖼️ Image: {len(r['images']['ids'])}个")
        for j, image_id in enumerate(r['images']['ids'][:2]):
            print(f"      [{j+1}] {image_id}")
        
        # 显示报告信息
        if 'report' in r:
            print(f"\n  📝 LLM生成报告:")
            report = r['report']
            if report.get('success'):
                report_data = report.get('report', {})
                print(f"      ✅ 生成成功")
                print(f"      📋 摘要: {report_data.get('summary', '无')[:200]}...")
                print(f"      📌 关键点: {len(report_data.get('key_points', []))}个")
                for kp in report_data.get('key_points', [])[:3]:
                    print(f"          - {kp}")
                print(f"      📄 内容章节: {len(report_data.get('content_sections', []))}个")
                for section in report_data.get('content_sections', [])[:2]:
                    print(f"          - {section.get('title', '未命名')}")
                print(f"      💻 代码说明: {len(report_data.get('code_explanations', []))}个")
                print(f"      📊 表格解读: {len(report_data.get('table_insights', []))}个")
                print(f"      🖼️ 图片描述: {len(report_data.get('image_descriptions', []))}个")
            else:
                print(f"      ❌ 生成失败: {report.get('error', '未知错误')}")
        
        print(f"\n  ⏱️ 耗时统计:")
        print(f"      阶段1 (Embedding): {r['timing']['stage1_embedding']:.2f}s")
        print(f"      阶段2 (Rerank): {r['timing']['stage2_rerank']:.2f}s")
        print(f"      阶段3 (扩散): {r['timing']['stage3_spread']:.2f}s")
        print(f"      阶段4 (最终Rerank): {r['timing']['stage4_final_rerank']:.2f}s")
        if 'stage5_report' in r['timing']:
            print(f"      阶段5 (报告生成): {r['timing']['stage5_report']:.2f}s")
        print(f"      总计: {r['timing']['total']:.2f}s")
    
    # 保存报告和前端Prompt到文件
    if results and 'report' in results[0]:
        import json
        output_dir = 'outputs/Harmony_docs_zh_cn'
        os.makedirs(output_dir, exist_ok=True)
        
        reports_to_save = []
        frontend_prompts = []
        
        for i, r in enumerate(results):
            if 'report' in r and r['report'].get('success'):
                # 保存报告数据
                reports_to_save.append({
                    'query': r['query'],
                    'report': r['report']['report'],
                    'chunks': r['chunks'],
                    'codes': r['codes'],
                    'tables': r['tables'],
                    'images': r['images']
                })
                
                # 保存前端Prompt
                frontend_prompts.append({
                    'query': r['query'],
                    'prompt': r['report'].get('frontend_prompt', '')
                })
                
                # 单独保存每个查询的前端Prompt为md文件
                prompt_file = os.path.join(output_dir, f'frontend_prompt_{i+1}.md')
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(r['report'].get('frontend_prompt', ''))
                print(f"  📝 前端Prompt已保存: {prompt_file}")
        
        # 保存汇总报告
        report_output_path = os.path.join(output_dir, 'reports.json')
        with open(report_output_path, 'w', encoding='utf-8') as f:
            json.dump(reports_to_save, f, ensure_ascii=False, indent=2)
        print(f"\n📁 报告JSON已保存到: {report_output_path}")
        
        # 保存所有前端Prompt汇总
        prompts_output_path = os.path.join(output_dir, 'frontend_prompts.json')
        with open(prompts_output_path, 'w', encoding='utf-8') as f:
            json.dump(frontend_prompts, f, ensure_ascii=False, indent=2)
        print(f"📁 前端Prompts汇总已保存到: {prompts_output_path}")

if __name__ == "__main__":
    main()
