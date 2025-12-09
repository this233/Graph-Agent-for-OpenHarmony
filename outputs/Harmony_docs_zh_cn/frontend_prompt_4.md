# 用户问题
What is OpenHarmony's development model?

# 核心回答
OpenHarmony采用组件化、分层化的开发模型，支持从128KiB到xGiB内存资源的全场景设备。其开发模型包含三个核心维度：系统类型划分、应用模型演进和开发方式选择。系统类型方面，OpenHarmony定义了轻量系统（面向MCU处理器，128KiB内存，适用于传感器、穿戴设备）、小型系统（面向应用处理器，1MiB内存，适用于IP Camera、路由器）和标准系统（面向应用处理器，128MiB内存，适用于高端显示屏）。应用模型方面，从API 7的FA模型演进到API 9的Stage模型，后者是目前主推的长期演进模型。开发方式上，提供基于IDE的图形化开发和基于命令行的传统开发两种选择，满足不同开发者的习惯需求。

## 关键要点
- 三种基础系统类型：轻量系统、小型系统、标准系统
- 两种应用模型：FA模型和Stage模型
- 组件化设计支持按需配置
- 支持两种开发方式：IDE和命令行
- 全场景分布式操作系统架构

# 详细内容

## 1. 系统类型分层模型

OpenHarmony的系统类型分层模型是其开发架构的核心基础，通过定义三种基础系统类型来适应不同硬件能力的设备。轻量系统面向MCU类处理器，如Arm Cortex-M、RISC-V 32位设备，这些设备硬件资源极其有限，最小内存要求仅为128KiB。该类型系统提供多种轻量级网络协议、轻量级图形框架以及丰富的IOT总线读写部件，主要支撑智能家居领域的连接类模组、传感器设备、穿戴类设备等资源受限场景。小型系统面向应用处理器，如Arm Cortex-A设备，最小内存要求为1MiB，相比轻量系统提供更高的安全能力、标准图形框架和视频编解码的多媒体能力，适用于智能家居领域的IP Camera、电子猫眼、路由器以及智慧出行域的行车记录仪等设备。标准系统同样面向应用处理器，但硬件要求更高，最小内存为128MiB，提供增强的交互能力、3D GPU及硬件合成能力、更多控件及动效更丰富的图形能力，以及完整的应用框架，主要支撑高端冰箱显示屏等需要丰富交互体验的产品。这种分层设计使得设备开发者能够基于目标硬件能力自由选择系统组件进行集成，实现最小系统的开发。

**本节要点：**
- 按硬件能力划分三种系统类型
- 支持128KiB到128MiB内存范围
- 针对不同应用场景优化系统能力

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/device-dev-guide.md`
  定位: 1 导读 > 1.1 系统类型
  相关性: 详细定义了三种基础系统类型的硬件要求、处理器类型和适用场景
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/quickstart-overview.md`
  定位: 1 快速入门概述
  相关性: 通过表格形式清晰对比三种系统类型的关键参数

## 2. 应用模型演进

OpenHarmony的应用模型是开发者进行应用程序开发的重要抽象层，提供了应用程序必备的组件和运行机制。随着系统演进，OpenHarmony先后提供了两种应用模型：FA模型和Stage模型。FA模型从OpenHarmony API 7开始支持，是早期的应用模型，但目前已不再主推。Stage模型从OpenHarmony API 9开始新增，是目前主推且会长期演进的模型。Stage模型之所以得名，是因为它提供了AbilityStage、WindowStage等类作为应用组件和Window窗口的'舞台'。这种设计使得应用开发更加统一和高效，开发者可以基于一套统一的模型进行应用开发。在Stage模型中，应用组件和窗口管理更加规范，提供了更好的开发体验和系统稳定性。快速入门指导目前都以Stage模型为例提供开发指导，体现了OpenHarmony对Stage模型的重视和推广力度。应用模型的演进反映了OpenHarmony在架构设计上的持续优化，从早期的功能导向逐步转向更加规范和可扩展的架构设计。

**本节要点：**
- 从FA模型演进到Stage模型
- Stage模型是当前主推的长期演进模型
- 提供统一的开发框架和运行机制

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/quick-start/start-overview.md`
  定位: 1 开发准备 > 1.1 基本概念 > 1.1.2 应用模型
  相关性: 明确说明两种应用模型的演进历程和当前推荐使用Stage模型

## 3. 开发方式选择

OpenHarmony为开发者提供了两种主要的开发入门方式，以适应不同的开发习惯和需求。基于IDE的入门方式使用DevEco Device Tool作为开发工具，采用Windows+Ubuntu混合开发环境。在这种模式下，Windows主要用于代码开发、代码调试和烧录等操作，而Ubuntu环境则负责源码编译。IDE方式提供界面化的操作接口，为开发者提供更快捷的开发体验，特别适合不熟悉命令行操作或习惯界面化操作的开发者。基于命令行的入门方式则通过命令行工具包进行操作，在Linux系统中进行编译时，所有相关操作都通过命令实现，在Windows系统中使用开发板厂商提供的工具进行代码烧录。命令行方式提供了简便统一的工具链安装方式，适合习惯使用命令行操作的开发者。这两种开发方式都支持三种系统类型的开发，包括轻量系统基于Hi3861开发板、小型系统基于Hi3516开发板、标准系统基于RK3568开发板，涵盖了从环境搭建、代码编写、编译、烧录到运行的全流程指导。

**本节要点：**
- IDE方式适合图形化操作习惯
- 命令行方式适合传统开发习惯
- 支持完整的开发流程

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/quickstart-overview.md`
  定位: 1 快速入门概述 > 1.1 操作方式
  相关性: 详细对比了两种开发方式的特点、工具和适用人群

## 4. 组件化与系统能力

OpenHarmony采用组件化设计理念，支持设备开发者基于目标硬件能力自由选择系统组件进行集成。系统将可选的系统组件组合为一系列描述为特性或功能的系统能力（SysCap），方便设备开发者理解和选择。这种组件化设计使得OpenHarmony能够在128KiB到xGiB RAM资源的广泛设备上运行系统组件。在应用开发层面，OpenHarmony提供了两个重要框架：应用程序框架（应用模型）和UI框架。所有应用都应该在这两个框架的基础上进行功能开发。此外，系统还提供了丰富的功能开发指导，包括ArkTS语言基础类库、Web、通知、窗口管理、WebGL、媒体、安全、网络与连接、电话服务、数据管理、文件管理、任务管理、设备使用信息统计、DFX、国际化、应用测试、IDL工具以及Native API等相关指导。这种组件化设计不仅提高了系统的灵活性，还使得OpenHarmony能够更好地适应不同硬件平台和设备需求。

**本节要点：**
- 组件化设计支持按需配置
- 系统能力（SysCap）机制
- 提供完整的开发框架和工具链

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/device-dev-guide.md`
  定位: 1 导读 > 1.1 系统类型
  相关性: 说明组件化设计理念和系统能力机制
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/application-dev-guide-for-gitee.md`
  定位: 1 应用开发导读 > 1.2 开发
  相关性: 列出应用开发的核心框架和功能模块

# 相关图片

## 图片 1
![OpenHarmony部件化设计示意图，展示系统能力如何分解为独立部件并通过拼装配置适应不同设备需求](https://gitee.com/openharmony/docs/raw/master/zh-cn/design/figures/Component-Definition.png)

**说明：** OpenHarmony部件化设计示意图，展示系统能力如何分解为独立部件并通过拼装配置适应不同设备需求

**上下文：** 该图片出现在部件设计和开发指南中，用于说明OpenHarmony如何通过部件化设计解决硬件碎片化问题

**与问题的相关性：** 直观展示OpenHarmony组件化开发模型的核心设计理念

**尺寸：** 760x459像素
**建议显示方式：** inline

**来源文件：** `/root/code/docs/zh-cn/design/OpenHarmony部件设计和开发指南.md`
**定位：** 1 OpenHarmony部件设计和开发指南 > 1.1 基本概念 > 1.1.1 部件定义

# 相关表格

## 表格 1: 基础类型系统简介

**描述：** 详细对比OpenHarmony三种基础系统类型的硬件要求、处理器类型、系统能力和适用产品

| 类型 | 处理器 | 最小内存 | 能力 |
| :--------: | :--------: | -------- | -------- |
| 轻量系统（mini system） | MCU类处理器（例如Arm Cortex-M、RISC-V 32位的设备） | 128KiB | 提供多种轻量级网络协议，轻量级的图形框架，以及丰富的IOT总线读写部件等。可支撑的产品如智能家居领域的连接类模组、传感器设备、穿戴类设备等。 |
| 小型系统（small system） | 应用处理器（例如Arm Cortex-A的设备） | 1MiB | 提供更高的安全能力、标准的图形框架、视频编解码的多媒体能力。可支撑的产品如智能家居领域的IP Camera、电子猫眼、路由器以及智慧出行域的行车记录仪等。 |
| 标准系统（standard system） | 应用处理器（例如Arm Cortex-A的设备） | 128MiB | 提供增强的交互能力、3D GPU以及硬件合成能力、更多控件以及动效更丰富的图形能力、完整的应用框架。可支撑的产品如高端的冰箱显示屏。 |

**关键数据：**
- 轻量系统：MCU处理器，128KiB内存
- 小型系统：应用处理器，1MiB内存
- 标准系统：应用处理器，128MiB内存

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/quick-start/quickstart-overview.md`
**定位：** 1 快速入门概述

## 表格 2: 设备类型硬件与内核要求

**描述：** 展示不同设备类型的硬件内存要求和对应的支持内核

| 设备类型    | 硬件要求        | 支持的内核          |
|---------|-------------|----------------|
| 轻量系统类设备 | 内存>128KB    | LiteOS-M       |
| 小型系统类设备 | 内存>1MB、有MMU | LiteOS-A、Linux |
| 标准系统类设备 | 内存>128MB    |  Linux       |

**关键数据：**
- 轻量系统：LiteOS-M内核
- 小型系统：LiteOS-A或Linux内核
- 标准系统：Linux内核

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/device-dev/porting/Readme-CN.md`
**定位：** 1 概述

# 相关代码

## 代码 1: 系统能力（SysCap）使用示例

**功能说明：** 这段代码展示了OpenHarmony中系统能力（SysCap）的使用方式。通过canIUse接口在运行时查询设备是否具备某个特定的系统能力，如地理位置服务。如果设备支持该能力，则调用相应的API获取当前位置信息；如果不支持，则输出提示信息。这种机制确保了应用在不同设备上的兼容性，开发者可以根据设备实际能力动态调整应用行为。

**使用注意：** 需要在支持SysCap的设备上运行，确保导入正确的模块

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
- 组件化架构
- 系统能力（SysCap）
- AbilityStage
- WindowStage
- ArkTS语言

# 推荐阅读

- **应用模型详细解读**: 深入了解FA模型和Stage模型的架构差异、设计思想和适用场景
- **部件设计和开发指南**: 掌握OpenHarmony组件化设计的核心理念和实现方式
- **API治理章程**: 了解OpenHarmony不同API类型的管理规范和兼容性要求
