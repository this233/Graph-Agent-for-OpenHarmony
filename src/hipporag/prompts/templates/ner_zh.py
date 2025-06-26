ner_system = """您的任务是从给定的段落中提取与开源鸿蒙(OpenHarmony)相关的专有名词作为命名实体。
这些命名实体将用于构建开源鸿蒙知识图谱，助力开发者掌握鸿蒙技术栈、高效开发，参与开源社区共建。

请重点提取以下类型的开源鸿蒙相关专有名词：
1. 技术组件和框架：如分布式软总线、Ability框架、UI框架等
2. 系统架构层次：如内核层、系统服务层、框架层、应用层等
3. 系统类型：如轻量系统、小型系统、标准系统等
4. 硬件架构：如Arm Cortex-M、RISC-V、x86、MCU等
5. API接口和参数：如系统API、应用API、接口名称、方法名、参数名、返回值类型、回调函数、事件监听器等
6. 开发工具和环境：如开发工具链、编译脚本、配置文件、IDE工具等
7. 产品设备类型：如智能家居设备、IP Camera、路由器等
8. 组织和项目名称：如OpenHarmony、开放原子开源基金会等
9. 技术概念：如子系统、组件、分布式能力等

请用JSON列表格式回复实体，忽略通用词汇。
"""

one_shot_ner_paragraph = """OpenHarmony分布式软总线
OpenHarmony的分布式软总线是多设备终端的统一基座，为设备间的无缝互联提供了统一的分布式通信能力。
它基于Arm Cortex-A处理器架构，支持轻量系统和标准系统的部署。
分布式软总线能够快速发现并连接设备，与分布式数据管理、分布式任务调度共同构成OpenHarmony的核心分布式能力。
开发者可以通过Ability框架和UI框架实现一次开发、多端部署的应用程序。
开发时需要调用connectDevice()接口建立连接，通过onDeviceStateChange回调函数监听设备状态变化，
deviceId参数用于标识目标设备，ConnectionCallback接口处理连接结果。"""

one_shot_ner_output = """{"named_entities":
    ["OpenHarmony", "分布式软总线", "Arm Cortex-A", "轻量系统", "标准系统", "分布式数据管理", "分布式任务调度", "分布式能力", "Ability框架", "UI框架", "connectDevice", "onDeviceStateChange", "deviceId", "ConnectionCallback"]
}
"""

prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"}
] 