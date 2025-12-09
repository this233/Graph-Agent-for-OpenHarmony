# 用户问题
What is OpenHarmony's development framework?

# 核心回答
OpenHarmony的开发框架是一个分层、模块化的架构体系，旨在支持从轻量级到标准系统的全场景设备部署。该框架基于三大核心理念：硬件互助资源共享、一次开发多端部署、统一OS弹性部署。在技术架构上，OpenHarmony采用四层设计：内核层支持LiteOS和Linux多内核；系统服务层提供分布式能力；框架层包含ArkUI框架和用户程序框架；应用层支持各类应用开发。ArkUI框架提供声明式和类Web两种开发范式，应用模型则包括主推的Stage模型和传统的FA模型。这种设计使得开发者能够基于统一的框架进行应用开发，同时确保应用在不同设备类型上的兼容性和性能优化。

## 关键要点
- 分层架构设计：内核层、系统服务层、框架层、应用层
- ArkUI框架提供声明式和类Web两种开发范式
- 应用模型包括Stage模型和FA模型
- 支持三种基础系统类型：轻量系统、小型系统、标准系统
- 分布式能力：软总线、数据管理、任务调度、设备虚拟化

# 详细内容

## 1. 整体架构设计

OpenHarmony的开发框架采用分层架构设计，从下到上分为四个主要层次：内核层、系统服务层、框架层和应用层。内核层支持多内核设计，包括LiteOS和Linux，其中LiteOS-M仅支持轻量系统，LiteOS-A仅支持小型系统，而Linux同时支持小型系统和标准系统。这种多内核支持使得OpenHarmony能够适应不同硬件资源的设备需求。系统服务层进一步分为基本系统服务子系统和增强系统服务子系统集，为上层提供基础的系统服务能力。框架层是开发者直接接触的部分，包括ArkUI框架和用户程序框架，为应用开发提供必要的组件和运行机制。应用层则包含系统应用和第三方扩展应用。这种分层设计使得系统具有良好的模块化和可扩展性，开发者可以根据目标设备的硬件能力选择相应的组件进行集成，实现从128KiB到GiB级别RAM资源的弹性部署。整个架构支持ARM、RISC-V、x86等各种CPU架构，确保在不同硬件平台上的兼容性。

**本节要点：**
- 四层架构：内核层、系统服务层、框架层、应用层
- 多内核支持：LiteOS和Linux
- 弹性部署支持不同硬件资源

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/OpenHarmony-Overview_zh.md`
  定位: 1 OpenHarmony开源项目 > 1.3 技术特性
  相关性: 描述了OpenHarmony的整体技术特性和架构设计理念
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/figures/1.png`
  定位: 1 OpenHarmony开源项目 > 1.2 技术架构
  相关性: 展示了OpenHarmony的技术架构层次图

## 2. ArkUI框架

ArkUI框架是OpenHarmony为开发者提供的UI开发框架，包含UI组件、动画、绘制、交互事件、JS API扩展机制等核心能力。该框架提供了两种开发范式：基于ArkTS的声明式开发范式和兼容JS的类Web开发范式。声明式开发范式使用ArkTS语言，采用数据驱动更新的方式，适用于复杂度较大、团队合作度较高的程序，主要面向移动系统应用开发人员和系统应用开发人员。类Web开发范式使用JS语言，同样采用数据驱动更新，适用于界面较为简单的程序应用和卡片，主要面向Web前端开发人员。两种开发范式的UI后端引擎和语言运行时是共用的，其中UI后端引擎实现了ArkUI框架的六种基本能力。声明式开发范式无需JS Framework进行页面DOM管理，渲染更新链路更为精简，占用内存更少，因此更推荐开发者选用声明式开发范式来搭建应用UI界面。对于不同系统类型，ArkUI框架有不同的实现：标准系统支持完整的声明式开发范式和类Web开发范式，而小型系统则通过ace_engine_lite实现了轻量级的ArkUI类Web开发范式lite版本。在某些配置较低的设备上进行系统应用开发时，还可以考虑选择C++ API，因为相比类Web范式，它具有更高的性能和更好的灵活性。

**本节要点：**
- 两种开发范式：声明式和类Web
- 声明式开发范式推荐用于标准系统
- 类Web开发范式lite用于小型系统

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/readme/ArkUI框架子系统.md`
  定位: 1 ArkUI框架子系统 > 1.1 简介
  相关性: 详细介绍了ArkUI框架的结构和两种开发范式
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/quick-start/start-overview.md`
  定位: 1 开发准备 > 1.1 基本概念 > 1.1.1 UI框架
  相关性: 提供了UI框架的对比和使用场景说明

## 3. 应用模型框架

OpenHarmony的应用模型框架为开发者提供了应用程序所需能力的抽象提炼，包含应用程序必备的组件和运行机制。随着系统的演进发展，OpenHarmony先后提供了两种应用模型：Stage模型和FA模型。Stage模型是OpenHarmony API 9开始新增的模型，是目前主推且会长期演进的模型。在该模型中，由于提供了AbilityStage、WindowStage等类作为应用组件和Window窗口的'舞台'，因此称这种应用模型为Stage模型。Stage模型开发是快速入门的主要示例，提供了完整的开发指导。FA模型是OpenHarmony API 7开始支持的模型，已经不再主推，快速入门章节不再对此展开提供开发指导。应用模型框架的设计使得开发者可以基于一套统一的模型进行应用开发，使应用开发更简单、高效。在最新的OpenHarmony 5.0.1版本中，应用框架还新增了ArkTS和C API，用于支持创建应用子进程的能力，新增C API用于获取应用的信息如缓存路径、文件加密模式、包名等，并支持设置应用级别的字体放大倍数。这些能力的增强进一步丰富了应用开发的可能性。

**本节要点：**
- 两种应用模型：Stage模型和FA模型
- Stage模型是主推模型
- 提供统一的开发模型简化开发流程

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/quick-start/start-overview.md`
  定位: 1 开发准备 > 1.1 基本概念 > 1.1.2 应用模型
  相关性: 详细介绍了OpenHarmony的两种应用模型及其特点
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/release-notes/OpenHarmony-v5.0.1-release.md`
  定位: 1 OpenHarmony 5.0.1 Release > 1.1 版本概述 > 1.1.1 应用框架
  相关性: 提供了应用框架在最新版本中的增强功能

## 4. 分布式能力框架

OpenHarmony的分布式能力框架是实现'硬件互助，资源共享'核心理念的关键组成部分，主要包括四个核心模块：分布式软总线、分布式数据管理、分布式任务调度和设备虚拟化。分布式软总线是多设备终端的统一基座，为设备间的无缝互联提供了统一的分布式通信能力，能够快速发现并连接设备，高效地传输任务和数据。分布式数据管理基于分布式软总线，实现了应用程序数据和用户数据的分布式管理，使得用户数据不再与单一物理设备绑定，业务逻辑与数据存储分离，应用跨设备运行时数据无缝衔接，为打造一致、流畅的用户体验创造了基础条件。分布式任务调度基于分布式软总线、分布式数据管理、分布式Profile等技术特性，构建统一的分布式服务管理机制，支持对跨设备的应用进行远程启动、远程调用、绑定/解绑以及迁移等操作，能够根据不同设备的能力、位置、业务运行状态、资源使用情况并结合用户的习惯和意图，选择最合适的设备运行分布式任务。设备虚拟化平台可以实现不同设备的资源融合、设备管理、数据处理，将周边设备作为手机能力的延伸，共同形成一个超级虚拟终端。这些分布式能力共同构成了OpenHarmony在多设备协同方面的核心竞争力。

**本节要点：**
- 四大分布式模块：软总线、数据管理、任务调度、设备虚拟化
- 支持跨设备应用运行和数据同步
- 构建超级虚拟终端概念

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/OpenHarmony-Overview_zh.md`
  定位: 1 OpenHarmony开源项目 > 1.3 技术特性
  相关性: 详细描述了分布式能力的各个模块及其功能

## 5. 多端部署与系统能力

OpenHarmony通过'一次开发，多端部署'的设计理念，提供了用户程序框架、Ability框架以及UI框架，能够保证开发的应用在多终端运行时保证一致性。多终端软件平台API具备一致性，确保用户程序的运行兼容性。OpenHarmony定义了三种基础系统类型：轻量系统面向MCU类处理器，最小内存128KiB，提供轻量级网络协议和图形框架，适用于智能家居连接类模组、传感器设备等；小型系统面向应用处理器，最小内存1MiB，提供更高的安全能力、标准图形框架和多媒体能力，适用于IP Camera、路由器等；标准系统同样面向应用处理器，最小内存128MiB，提供增强的交互能力、3D GPU和完整应用框架，适用于高端冰箱显示屏等。系统能力（SysCap）机制使得应用开发者可以通过CanIUse接口在运行时查询设备具备某个SysCap，从而保证应用在不同设备上的兼容性。此外，SDK中定义了典型设备类型包含的SysCap集合，应用开发时选定设备类型的情况下，如果API调用了超出设备类型必选SysCap对应的API范围，IDE会提示API不被该设备支持，导致应用编译失败。这种机制确保了应用在不同设备类型上的适配性和兼容性。

**本节要点：**
- 三种基础系统类型：轻量、小型、标准系统
- 系统能力（SysCap）机制确保兼容性
- 支持在开发过程中预览终端能力适配情况

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/quickstart-overview.md`
  定位: 1 快速入门概述
  相关性: 提供了三种基础系统类型的详细定义和适用场景
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/design/OpenHarmony部件设计和开发指南.md`
  定位: 1 OpenHarmony部件设计和开发指南 > 1.2 部件设计 > 1.2.3 系统能力（SysCap） > 1.2.3.1 应用开发者如何使用SysCap?
  相关性: 展示了SysCap的使用方法和代码示例

# 相关图片

## 图片 1
![OpenHarmony技术架构图，展示从内核层到应用层的四层设计结构](https://gitee.com/openharmony/docs/raw/master/zh-cn/figures/1.png)

**说明：** OpenHarmony技术架构图，展示从内核层到应用层的四层设计结构

**上下文：** 该图片出现在OpenHarmony概述文档的技术架构部分，用于直观展示系统的整体架构层次

**与问题的相关性：** 直接展示了OpenHarmony开发框架的整体架构设计，是理解框架层次的重要视觉参考

**尺寸：** 1783x866像素
**建议显示方式：** full-width

**来源文件：** `/root/code/docs/zh-cn/OpenHarmony-Overview_zh.md`
**定位：** 1 OpenHarmony开源项目 > 1.2 技术架构

## 图片 2
![ArkUI框架结构图，展示声明式开发范式和类Web开发范式的架构关系](https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/ui/figures/arkui-framework.png)

**说明：** ArkUI框架结构图，展示声明式开发范式和类Web开发范式的架构关系

**上下文：** 该图片出现在ArkUI框架子系统介绍中，用于说明两种开发范式的技术实现差异

**与问题的相关性：** 展示了UI框架的核心架构，帮助理解不同开发范式的技术实现

**尺寸：** 587x451像素
**建议显示方式：** inline

**来源文件：** `/root/code/docs/zh-cn/readme/ArkUI框架子系统.md`
**定位：** 1 ArkUI框架子系统 > 1.1 简介

# 相关表格

## 表格 1: 基础类型系统简介

**描述：** 展示OpenHarmony三种基础系统类型的硬件要求和适用场景

| 类型 | 处理器 | 最小内存 | 能力 |
| :--------: | :--------: | -------- | -------- |
| 轻量系统（mini&nbsp;system） | MCU类处理器（例如Arm&nbsp;Cortex-M、RISC-V&nbsp;32位的设备） | 128KiB | 提供多种轻量级网络协议，轻量级的图形框架，以及丰富的IOT总线读写部件等。可支撑的产品如智能家居领域的连接类模组、传感器设备、穿戴类设备等。 |
| 小型系统（small&nbsp;system） | 应用处理器（例如Arm&nbsp;Cortex-A的设备） | 1MiB | 提供更高的安全能力、标准的图形框架、视频编解码的多媒体能力。可支撑的产品如智能家居领域的IP&nbsp;Camera、电子猫眼、路由器以及智慧出行域的行车记录仪等。 |
| 标准系统（standard&nbsp;system） | 应用处理器（例如Arm&nbsp;Cortex-A的设备） | 128MiB | 提供增强的交互能力、3D&nbsp;GPU以及硬件合成能力、更多控件以及动效更丰富的图形能力、完整的应用框架。可支撑的产品如高端的冰箱显示屏。 |

**关键数据：**
- 轻量系统：MCU处理器，128KiB内存
- 小型系统：应用处理器，1MiB内存
- 标准系统：应用处理器，128MiB内存

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/quickstart-overview.md`
**定位：** 1 快速入门概述

## 表格 2: UI框架开发范式对比

**描述：** 对比ArkUI框架的两种开发范式在语言生态、UI更新方式和适用场景等方面的差异

| 开发范式名称 | 语言生态 | UI更新方式 | 适用场景 | 适用人群 |
| ---------------- | ------------ | -------------- | -------------------------------- | -------------------------------------- |
| 声明式开发范式 | ArkTS语言 | 数据驱动更新 | 复杂度较大、团队合作度较高的程序 | 移动系统应用开发人员、系统应用开发人员 |
| 类Web开发范式 | JS语言 | 数据驱动更新 | 界面较为简单的程序应用和卡片 | Web前端开发人员 |

**关键数据：**
- 声明式开发范式：ArkTS语言，数据驱动更新
- 类Web开发范式：JS语言，数据驱动更新
- 声明式适用于复杂程序，类Web适用于简单界面

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/quick-start/start-overview.md`
**定位：** 1 开发准备 > 1.1 基本概念 > 1.1.1 UI框架

# 相关代码

## 代码 1: SysCap运行时检测示例

**功能说明：** 这段代码展示了在OpenHarmony中如何使用SysCap（系统能力）机制进行运行时检测。通过canIUse接口查询设备是否具备地理位置服务能力，如果设备支持则获取当前位置信息，否则输出提示信息。这种机制确保了应用在不同设备上的兼容性，避免在不支持特定功能的设备上调用相关API导致错误。

**使用注意：** 需要在支持SysCap的OpenHarmony版本中使用，确保导入正确的模块

**涉及API：** canIUse, geolocation.getCurrentLocation

```javascript
import geolocation from '@ohos.geolocation'

const isLocationAvailable = canIUse('SystemCapability.Location.Location');
if (isLocationAvailable) {
geolocation.getCurrentLocation((location) => {
console.log(location.latitude, location.longitue);
});
} else {
console.log('Location not by this device.')
}
```

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/design/OpenHarmony部件设计和开发指南.md`
**定位：** 1 OpenHarmony部件设计和开发指南 > 1.2 部件设计 > 1.2.3 系统能力（SysCap） > 1.2.3.1 应用开发者如何使用SysCap?

# 相关概念

- 分布式操作系统
- 组件化设计
- 多内核架构
- 超级虚拟终端
- 系统能力（SysCap）
- 开发范式

# 推荐阅读

- **ArkUI框架详细开发指南**: 深入了解UI框架的具体使用方法和最佳实践
- **Stage模型开发概述**: 掌握主推应用模型的详细开发流程
- **分布式应用开发**: 学习如何利用分布式能力开发跨设备应用
- **系统能力开发指南**: 了解如何确保应用在不同设备上的兼容性
