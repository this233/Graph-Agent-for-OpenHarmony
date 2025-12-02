# -*- coding: utf-8 -*-
"""
Default DSPy prompt configuration for fact filtering in HippoRAG.

This module contains the best performing DSPy prompt configuration used for
filtering facts based on their relevance to a given query. The prompt is used
by the fact filtering component to select the most relevant facts from a
candidate list.

改进说明：
1. 将 "up to 4" 改为 "up to 10"，允许更多相关事实通过
2. 添加中英文双语支持说明
3. 优化 prompt，使其更适合技术文档检索场景
"""

# 改进版的 DSPy prompt 配置
best_dspy_prompt = {
    'prog': {
        'lm': None,
        'traces': [],
        'train': [],
        
        # Training/demonstration examples for the fact filtering task
        'demos': [
            {
                'augmented': True,
                'question': 'Are Imperial River (Florida) and Amaradia (Dolj) both located in the same country?',
                'fact_before_filter': '{"fact": [["imperial river", "is located in", "florida"], ["imperial river", "is a river in", "united states"], ["imperial river", "may refer to", "south america"], ["amaradia", "flows through", "ro ia de amaradia"], ["imperial river", "may refer to", "united states"]]}',
                'fact_after_filter': '{"fact":[["imperial river","is located in","florida"],["imperial river","is a river in","united states"],["amaradia","flows through","ro ia de amaradia"]]}'
            },
            {
                'augmented': True,
                'question': "When is the director of film The Ancestor 's birthday?",
                'fact_before_filter': '{"fact": [["jean jacques annaud", "born on", "1 october 1943"], ["tsui hark", "born on", "15 february 1950"], ["pablo trapero", "born on", "4 october 1971"], ["the ancestor", "directed by", "guido brignone"], ["benh zeitlin", "born on", "october 14  1982"]]}',
                'fact_after_filter': '{"fact":[["the ancestor","directed by","guido brignone"]]}'
            },
            {
                'augmented': True,
                'question': 'In what geographic region is the country where Teafuone is located?',
                'fact_before_filter': '{"fact": [["teafuaniua", "is on the", "east"], ["motuloa", "lies between", "teafuaniua"], ["motuloa", "lies between", "teafuanonu"], ["teafuone", "is", "islet"], ["teafuone", "located in", "nukufetau"]]}',
                'fact_after_filter': '{"fact":[["teafuone","is","islet"],["teafuone","located in","nukufetau"]]}'
            },
            {
                'augmented': True,
                'question': 'When did the director of film S.O.B. (Film) die?',
                'fact_before_filter': '{"fact": [["allan dwan", "died on", "28 december 1981"], ["s o b", "written and directed by", "blake edwards"], ["robert aldrich", "died on", "december 5  1983"], ["robert siodmak", "died on", "10 march 1973"], ["bernardo bertolucci", "died on", "26 november 2018"]]}',
                'fact_after_filter': '{"fact":[["s o b","written and directed by","blake edwards"]]}'
            },
            {
                'augmented': True,
                'question': 'Do both films: Gloria (1980 Film) and A New Life (Film) have the directors from the same country?',
                'fact_before_filter': '{"fact": [["sebasti n lelio watt", "received acclaim for directing", "gloria"], ["gloria", "is", "1980 american thriller crime drama film"], ["a brand new life", "is directed by", "ounie lecomte"], ["gloria", "written and directed by", "john cassavetes"], ["a new life", "directed by", "alan alda"]]}',
                'fact_after_filter': '{"fact":[["gloria","is","1980 american thriller crime drama film"],["gloria","written and directed by","john cassavetes"],["a new life","directed by","alan alda"]]}'
            },
            {
                'augmented': True,
                'question': 'What is the date of death of the director of film The Old Guard (1960 Film)?',
                'fact_before_filter': '{"fact": [["the old guard", "is", "1960 french comedy film"], ["gilles grangier", "directed", "the old guard"], ["the old guard", "directed by", "gilles grangier"], ["the old fritz", "directed by", "gerhard lamprecht"], ["oswald albert mitchell", "directed", "old mother riley series of films"]]}',
                'fact_after_filter': '{"fact":[["the old guard","is","1960 french comedy film"],["gilles grangier","directed","the old guard"],["the old guard","directed by","gilles grangier"]]}'
            },
            {
                'augmented': True,
                'question': "When is the composer of film Aulad (1968 Film) 's birthday?",
                'fact_before_filter': '{"fact": [["aulad", "has music composed by", "chitragupta shrivastava"], ["aadmi sadak ka", "has music by", "ravi"], ["ravi shankar sharma", "composed music for", "hindi films"], ["gulzar", "was born on", "18 august 1934"], ["aulad", "is a", "1968 hindi language drama film"]]}',
                'fact_after_filter': '{"fact":[["aulad","has music composed by","chitragupta shrivastava"],["aulad","is a","1968 hindi language drama film"]]}'
            },
            {
                'question': 'How many households were in the city where Angelical Tears located?',
                'fact_before_filter': '{"fact": [["dow city", "had", "219 households"], ["tucson", "had", "229 762 households"], ["atlantic city", "has", "15 504 households"], ["angelical tears", "located in", "oklahoma city"], ["atlantic city", "had", "15 848 households"]]}',
                'fact_after_filter': '{"fact": [["angelical tears", "located in", "oklahoma city"]]}'
            },
            {
                'question': "Did the movies In The Pope'S Eye and Virgin Mountain, originate from the same country?",
                'fact_before_filter': '{"fact": [["virgin mountain", "released in", "icelandic cinemas"], ["virgin mountain", "directed by", "dagur k ri"], ["virgin mountain", "icelandic title is", "f si"], ["virgin mountain", "won", "2015 nordic council film prize"], ["virgin mountain", "is a", "2015 icelandic drama film"]]}',
                'fact_after_filter': '{"fact": [["virgin mountain", "released in", "icelandic cinemas"], ["virgin mountain", "directed by", "dagur k ri"], ["virgin mountain", "icelandic title is", "f si"], ["virgin mountain", "won", "2015 nordic council film prize"], ["virgin mountain", "is a", "2015 icelandic drama film"]]}'
            },
            {
                'question': "Which film has the director who died earlier, The Virtuous Model or Bulldog Drummond'S Peril?",
                'fact_before_filter': '{"fact": [["the virtuous model", "is", "1919 american silent drama film"], ["bulldog drummond s peril", "directed by", "james p  hogan"], ["the virtuous model", "directed by", "albert capellani"], ["bulldog drummond s revenge", "directed by", "louis king"], ["bulldog drummond s peril", "is", "american film"]]}',
                'fact_after_filter': '{"fact": [["the virtuous model", "is", "1919 american silent drama film"], ["bulldog drummond s peril", "directed by", "james p  hogan"], ["the virtuous model", "directed by", "albert capellani"], ["bulldog drummond s peril", "is", "american film"]]}'
            },
            # 新增中文技术文档示例
            {
                'augmented': True,
                'question': 'OpenHarmony的系统架构是什么？',
                'fact_before_filter': '{"fact": [["OpenHarmony", "是", "开源分布式操作系统"], ["OpenHarmony", "采用分层设计", "内核层"], ["OpenHarmony", "采用分层设计", "框架层"], ["OpenHarmony", "采用分层设计", "应用层"], ["OpenHarmony", "支持", "多种设备类型"], ["本文档", "适用于", "OpenHarmony"]]}',
                'fact_after_filter': '{"fact": [["OpenHarmony", "是", "开源分布式操作系统"], ["OpenHarmony", "采用分层设计", "内核层"], ["OpenHarmony", "采用分层设计", "框架层"], ["OpenHarmony", "采用分层设计", "应用层"], ["OpenHarmony", "支持", "多种设备类型"]]}'
            },
            {
                'augmented': True,
                'question': 'What are the main features of OpenHarmony?',
                'fact_before_filter': '{"fact": [["OpenHarmony", "目标是", "面向全场景"], ["OpenHarmony", "支持", "分布式软总线"], ["OpenHarmony", "提供", "一次开发多端部署"], ["OpenHarmony", "是", "开源项目"], ["分布式特性", "是", "OpenHarmony关键特性"], ["第三方", "未经许可不得使用", "OpenHarmony标志"]]}',
                'fact_after_filter': '{"fact": [["OpenHarmony", "目标是", "面向全场景"], ["OpenHarmony", "支持", "分布式软总线"], ["OpenHarmony", "提供", "一次开发多端部署"], ["分布式特性", "是", "OpenHarmony关键特性"]]}'
            }
        ],
        
        # Signature configuration for the fact filtering task
        'signature': {
            'instructions': (
                'You are a critical component of a high-stakes question-answering system. '
                'Your task is to filter facts based on their relevance to a given query. '
                'The query may be in English or Chinese, and requires careful analysis and possibly multi-hop reasoning. '
                '\n\n'
                'Selection criteria:\n'
                '1. Select facts that directly relate to the query subject or its attributes\n'
                '2. Include facts that help establish connections for multi-hop reasoning\n'
                '3. Prefer facts with specific information over general statements\n'
                '4. Include definition/description facts about key entities mentioned in the query\n'
                '\n'
                'You must select up to 10 relevant facts from the provided candidate list. '
                'The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}. '
                'If no facts are relevant, return {"fact": []}. '
                'You must only use facts from the candidate list and not generate new facts.'
            ),
            'fields': [
                {'prefix': 'Question:', 'description': 'Query for retrieval (may be in English or Chinese)'},
                {'prefix': 'Fact Before Filter:', 'description': 'Candidate facts to be filtered'},
                {'prefix': 'Fact After Filter:', 'description': 'Filtered facts in JSON format (up to 10 facts)'}
            ]
        },
        
        # System prompt template
        'system': (
            'Your input fields are:\n'
            '1. `question` (str): Query for retrieval (may be in English or Chinese)\n'
            '2. `fact_before_filter` (str): Candidate facts to be filtered\n\n'
            'Your output fields are:\n'
            '1. `fact_after_filter` (Fact): Filtered facts in JSON format\n\n'
            'All interactions will be structured in the following way, with the appropriate values filled in.\n\n'
            '[[ ## question ## ]]\n'
            '{question}\n\n'
            '[[ ## fact_before_filter ## ]]\n'
            '{fact_before_filter}\n\n'
            '[[ ## fact_after_filter ## ]]\n'
            '{fact_after_filter}        # note: the value you produce must be parseable according to the following JSON schema: '
            '{"type": "object", "properties": {"fact": {"type": "array", "description": "A list of facts, each fact is a list of 3 strings: [subject, predicate, object]", "items": {"type": "array", "items": {"type": "string"}}, "title": "Fact"}}, "required": ["fact"], "title": "Fact"}\n\n'
            '[[ ## completed ## ]]\n\n'
            'In adhering to this structure, your objective is:\n'
            'You are a critical component of a high-stakes question-answering system. '
            'Your task is to filter facts based on their relevance to a given query. '
            'The query may be in English or Chinese, and requires careful analysis and possibly multi-hop reasoning.\n\n'
            'Selection criteria:\n'
            '1. Select facts that directly relate to the query subject or its attributes\n'
            '2. Include facts that help establish connections for multi-hop reasoning\n'
            '3. Prefer facts with specific information over general statements\n'
            '4. Include definition/description facts about key entities mentioned in the query\n\n'
            'You must select up to 10 relevant facts from the provided candidate list. '
            'The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}. '
            'If no facts are relevant, return {"fact": []}. '
            'You must only use facts from the candidate list and not generate new facts.'
        )
    }
}
