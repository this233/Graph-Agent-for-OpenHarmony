# 用户问题
What is OpenHarmony's development tools?

# 核心回答
OpenHarmony提供了一套完整的开发工具链，主要包括HUAWEI DevEco Studio和HUAWEI DevEco Device Tool两大核心工具。DevEco Studio是专门为OpenHarmony应用开发设计的集成开发环境，支持多种编程语言和框架，提供代码编辑、编译、调试、打包等全流程开发功能。DevEco Device Tool则是针对智能设备开发的集成开发环境，支持设备固件开发、烧录和调试。此外，OpenHarmony还提供Public SDK，这是面向应用开发者的软件开发工具包，不包含需要系统权限的系统接口，通过DevEco Studio默认获取。这些工具在不同OpenHarmony版本中都有相应的配套版本，从3.0系列到最新的6.0版本都保持工具链的持续更新和完善。

## 关键要点
- HUAWEI DevEco Studio是OpenHarmony应用开发推荐使用的IDE
- HUAWEI DevEco Device Tool是智能设备集成开发环境
- Public SDK是面向应用开发者的软件开发工具包
- 开发工具与OpenHarmony各版本保持配套关系
- 工具支持多种调试方式包括USB和无线调试

# 详细内容

## 1. 核心开发工具概述

OpenHarmony的开发工具生态系统主要由两大核心工具构成：HUAWEI DevEco Studio和HUAWEI DevEco Device Tool。DevEco Studio是专门为OpenHarmony应用开发设计的集成开发环境，它支持ArkTS、JavaScript等多种编程语言，提供完整的代码编辑、编译构建、调试测试、应用打包和发布功能。该工具支持跨平台开发，提供Windows和Mac版本，并且针对不同芯片架构（如X86和ARM）都有相应的优化版本。DevEco Studio还集成了丰富的模板和示例代码，帮助开发者快速上手OpenHarmony应用开发。DevEco Device Tool则是专门针对智能设备硬件开发的集成开发环境，支持设备固件开发、烧录、调试和测试等全流程开发工作。这两个工具都是可选但强烈推荐的，它们与OpenHarmony的各个版本保持严格的配套关系，确保开发环境的稳定性和兼容性。从OpenHarmony 3.0系列到最新的6.0版本，这些工具都在持续迭代更新，以适应新的API特性和开发需求。

**本节要点：**
- DevEco Studio专注于应用开发
- DevEco Device Tool专注于设备开发
- 工具与版本保持严格配套
- 支持跨平台开发

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/release-notes/OpenHarmony-v6.0-release.md`
  定位: 1.3 配套关系
  相关性: 详细说明了6.0版本的开发工具配套信息
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/release-notes/OpenHarmony-v5.0.0-release.md`
  定位: 1.2 配套关系
  相关性: 提供了5.0.0版本的开发工具版本和获取方式

## 2. HUAWEI DevEco Studio详细功能

HUAWEI DevEco Studio作为OpenHarmony应用开发的主要IDE，提供了丰富的开发功能。在代码编辑方面，它支持语法高亮、代码自动补全、错误检查、重构等功能，大大提高了开发效率。在编译构建方面，DevEco Studio支持增量编译和热重载，能够快速看到代码修改的效果。在调试功能方面，它支持多种调试方式，包括通过DevEco Studio内置调试器进行调试，或者使用hdc工具进行命令行调试。对于Web开发，DevEco Studio还支持使用DevTools工具进行前端页面调试，包括USB连接调试和无线调试两种模式。在应用打包方面，DevEco Studio能够将应用代码编译打包成HAP（Harmony Ability Package）格式，这是OpenHarmony应用的安装包格式。DevEco Studio还提供了项目管理、依赖管理、版本控制集成等辅助功能，为开发者提供一站式的开发体验。不同版本的OpenHarmony对应不同版本的DevEco Studio，如6.0 Release对应DevEco Studio 6.0.0 Release，5.0.0 Release对应DevEco Studio 5.0.0 Release等，确保开发工具与系统版本的兼容性。

**本节要点：**
- 支持代码编辑和自动补全
- 提供多种调试方式
- 支持HAP打包
- 版本配套确保兼容性

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/release-notes/OpenHarmony-v4.1.2-release.md`
  定位: 1.2 配套关系
  相关性: 展示了DevEco Studio的具体下载链接和不同平台版本
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/quick-start/hap-package.md`
  定位: 1.5 调试
  相关性: 详细说明了使用DevEco Studio进行调试的方法

## 3. HUAWEI DevEco Device Tool功能详解

HUAWEI DevEco Device Tool是专门为OpenHarmony智能设备开发设计的集成开发环境，主要面向硬件开发者和设备厂商。该工具支持设备固件的开发、编译、烧录和调试，提供完整的设备开发生命周期管理。DevEco Device Tool支持多种硬件平台和设备类型，能够与不同的开发板和芯片进行适配。在设备调试方面，它提供了丰富的调试工具和接口，帮助开发者快速定位和解决设备层面的问题。从版本演进来看，DevEco Device Tool从3.0 Release版本发展到4.0 Release版本，功能不断完善和增强。该工具通常与DevEco Studio配合使用，形成从应用到设备的完整开发解决方案。获取方式主要通过官方网站的下载页面，确保开发者能够获得最新稳定版本的工具。DevEco Device Tool的持续更新反映了OpenHarmony在物联网和智能设备领域的发展战略，为开发者提供强大的设备开发能力支持。

**本节要点：**
- 专注于设备固件开发
- 支持多种硬件平台
- 提供设备调试工具
- 与DevEco Studio形成完整解决方案

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/release-notes/OpenHarmony-v3.2.4-release.md`
  定位: 1.2 配套关系
  相关性: 说明了DevEco Device Tool在设备开发中的重要作用
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/release-notes/OpenHarmony-v5.0.0-release.md`
  定位: 1.2 配套关系
  相关性: 提供了DevEco Device Tool的具体获取链接

## 4. Public SDK与开发工具集成

Public SDK是OpenHarmony为应用开发者提供的软件开发工具包，它包含了开发OpenHarmony应用所需的各种API、库文件、工具和文档。Public SDK的一个重要特点是它不包含需要使用系统权限的系统接口，这确保了应用的安全性和稳定性。通过DevEco Studio默认获取的SDK就是Public SDK，这种集成方式简化了开发环境的配置过程。不同版本的OpenHarmony对应不同版本的Public SDK，如6.0 Release对应Ohos_sdk_public 6.0.0.47（API Version 20 Release），5.0.0 Release对应Ohos_sdk_public 5.0.0.71（API Version 12 Release）等。Public SDK的版本号通常包含API版本信息，帮助开发者了解所使用API的兼容性范围。除了Public SDK，OpenHarmony还提供Full SDK，但Full SDK需要特殊的权限和配置才能使用，主要面向系统开发者和设备厂商。对于大多数应用开发者来说，Public SDK已经能够满足开发需求，而且更加安全可靠。

**本节要点：**
- Public SDK面向应用开发者
- 不包含系统权限接口
- 通过DevEco Studio默认获取
- 版本与OpenHarmony保持同步

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/release-notes/OpenHarmony-v6.0-beta1.md`
  定位: 1.2 配套关系
  相关性: 详细说明了Public SDK的版本信息和获取方式
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/release-notes/OpenHarmony-v4.1-beta1.md`
  定位: 1.2 配套关系
  相关性: 展示了Public SDK在不同版本中的具体配置

## 5. 调试工具与方法

OpenHarmony提供了多种调试工具和方法来支持开发过程中的问题定位和解决。主要的调试工具包括DevEco Studio内置调试器、hdc工具和DevTools。DevEco Studio内置调试器提供图形化的调试界面，支持断点设置、变量查看、单步执行等传统调试功能。hdc（HarmonyOS Device Connector）是一个命令行工具，用于与OpenHarmony设备进行通信和调试，支持安装、卸载HAP包，执行shell命令等功能。DevTools则是专门用于Web前端页面调试的工具，支持USB连接调试和无线调试两种模式。在USB调试模式下，开发者需要通过代码调用setWebDebuggingAccess(true)接口开启Web调试开关。在无线调试模式下，除了开启调试开关外，还需要设置TCP Socket端口号。所有调试功能都需要在应用的配置文件中声明相应的权限，如INTERNET权限。这些调试工具和方法构成了完整的调试体系，能够满足不同场景下的调试需求，提高开发效率和质量。

**本节要点：**
- 提供多种调试工具
- 支持USB和无线调试
- 需要配置相应权限
- 形成完整调试体系

**来源：**
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/quick-start/hap-package.md`
  定位: 1.5 调试
  相关性: 详细说明了使用hdc工具进行调试的具体命令和方法
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/web/web-debugging-with-devtools.md`
  定位: 1.1 无线调试
  相关性: 展示了Web无线调试的具体代码实现
- 文件: `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/web/web-debugging-with-devtools.md`
  定位: 1.2 USB连接调试
  相关性: 说明了USB连接调试的配置方法

# 相关图片

## 图片 1
![DevEco Studio界面展示，显示项目结构和设备连接状态，表明开发环境已正确配置并可以开始开发工作](https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/device/driver/figures/device-connected.png)

**说明：** DevEco Studio界面展示，显示项目结构和设备连接状态，表明开发环境已正确配置并可以开始开发工作

**上下文：** 在环境准备章节中，用于展示DevEco Studio中设备连接成功的界面状态

**与问题的相关性：** 直观展示了DevEco Studio的开发环境和设备连接功能

**尺寸：** 1920x550像素
**建议显示方式：** full-width

**来源文件：** `/root/code/docs/zh-cn/application-dev/device/driver/environmental-preparation.md`
**定位：** 1.3 检验环境是否搭建成功

## 图片 2
![DevEco Studio中通过Tools菜单打开OpenHarmony SDK Manager的界面](https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/faqs/figures/zh-cn_image_0000001655128939.png)

**说明：** DevEco Studio中通过Tools菜单打开OpenHarmony SDK Manager的界面

**上下文：** 在如何替换full-SDK的指南中，展示查看本地SDK路径的操作步骤

**与问题的相关性：** 展示了DevEco Studio中管理SDK的工具界面

**尺寸：** 737x327像素
**建议显示方式：** inline

**来源文件：** `/root/code/docs/zh-cn/application-dev/faqs/full-sdk-switch-guide.md`
**定位：** 1.2 查看本地SDK路径

## 图片 3
![DevEco Studio中SDK Manager界面，显示本地OpenHarmony SDK安装路径和版本信息](https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/faqs/figures/zh-cn_image_0000001655128998.png)

**说明：** DevEco Studio中SDK Manager界面，显示本地OpenHarmony SDK安装路径和版本信息

**上下文：** 在SDK替换指南中，用于展示如何查看和管理本地SDK配置

**与问题的相关性：** 详细展示了DevEco Studio中SDK管理的具体界面和配置信息

**尺寸：** 981x707像素
**建议显示方式：** full-width

**来源文件：** `/root/code/docs/zh-cn/application-dev/faqs/full-sdk-switch-guide.md`
**定位：** 1.2 查看本地SDK路径

# 相关表格

## 表格 1: OpenHarmony 6.0 Release版本配套关系

**描述：** 展示了OpenHarmony 6.0 Release版本与相关开发工具的配套信息，包括SDK和IDE版本

| 软件 | 版本 | 备注 |
| -------- | -------- | -------- |
| OpenHarmony | 6.0 Release | NA |
| Public SDK | Ohos_sdk_public 6.0.0.47 (API Version 20 Release) | 面向应用开发者提供，不包含需要使用系统权限的系统接口。通过DevEco Studio默认获取的SDK为Public SDK。 |
| HUAWEI DevEco Studio（可选） | 6.0.0 Release | OpenHarmony应用开发推荐使用。<br />*待发布后提供*。 |
| HUAWEI DevEco Device Tool（可选） | 4.0 Release | OpenHarmony智能设备集成开发环境推荐使用。<br />[请点击这里获取](https://device.harmonyos.com/cn/develop/ide#download)。 |

**关键数据：**
- Public SDK版本：Ohos_sdk_public 6.0.0.47
- API版本：20 Release
- DevEco Studio版本：6.0.0 Release
- DevEco Device Tool版本：4.0 Release

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/release-notes/OpenHarmony-v6.0-release.md`
**定位：** 1.3 配套关系

## 表格 2: OpenHarmony 5.0.0 Release版本配套关系

**描述：** 展示了OpenHarmony 5.0.0 Release版本与开发工具的配套信息

| 软件 | 版本 | 备注 |
| -------- | -------- | -------- |
| OpenHarmony | 5.0.0 Release | NA |
| Public SDK | Ohos_sdk_public 5.0.0.71 (API Version 12 Release) | 面向应用开发者提供，不包含需要使用系统权限的系统接口。 |
| HUAWEI DevEco Studio（可选） | 5.0.0 Release | OpenHarmony应用开发推荐使用。<br />[请点击这里获取](https://developer.huawei.com/consumer/cn/download/)。 |
| HUAWEI DevEco Device Tool（可选） | 4.0 Release | OpenHarmony智能设备集成开发环境推荐使用。<br />[请点击这里获取](https://device.harmonyos.com/cn/develop/ide#download)。 |

**关键数据：**
- Public SDK版本：Ohos_sdk_public 5.0.0.71
- API版本：12 Release
- DevEco Studio版本：5.0.0 Release
- DevEco Device Tool版本：4.0 Release

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/release-notes/OpenHarmony-v5.0.0-release.md`
**定位：** 1.2 配套关系

# 相关代码

## 代码 1: Web调试开关配置代码

**功能说明：** 这段代码展示了如何在OpenHarmony应用中开启Web调试功能。通过调用setWebDebuggingAccess(true)接口，启用WebView的调试模式。这是使用DevTools进行Web页面调试的前提条件，如果没有开启调试开关，DevTools将无法发现和调试网页内容。代码使用了ArkTS语言，这是OpenHarmony推荐的应用开发语言。

**使用注意：** 需要在DevEco Studio应用工程hap模块的module.json5文件中增加INTERNET权限

**涉及API：** setWebDebuggingAccess(), WebviewController

```TypeScript
import { webview } from '@kit.ArkWeb';

@Entry
@Component
struct WebComponent {
controller: webview.WebviewController = new webview.WebviewController();

aboutToAppear() {
// 配置Web开启调试模式
webview.WebviewController.setWebDebuggingAccess(true);
}

build() {
Column() {
Web({ src: 'www.example.com', controller: this.controller })
}
}
```

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/web/web-debugging-with-devtools.md`
**定位：** 1.2 USB连接调试 > 1.2.1 应用代码开启Web调试开关

## 代码 2: 无线Web调试配置代码

**功能说明：** 这段代码演示了如何在OpenHarmony应用中启用WebView无线调试功能。与USB调试不同，无线调试需要指定TCP Socket端口号，允许开发者通过网络连接进行远程调试。代码包含了异常处理机制，确保在端口被占用或其他问题时能够正确处理。

**使用注意：** 需要确保指定的端口号可以被应用使用，避免端口冲突

**涉及API：** setWebDebuggingAccess(), WebviewController, BusinessError

```TypeScript
import { webview } from '@kit.ArkWeb';
import { BusinessError } from '@kit.BasicServicesKit';
const DEBUGGING_PORT: number = 8888;

@Entry
@Component
struct WebComponent {
controller: webview.WebviewController = new webview.WebviewController();

aboutToAppear(): void {
try {
// 配置Web开启无线调试模式，指定TCP Socket的端口。
webview.WebviewController.setWebDebuggingAccess(true, DEBUGGING_PORT);
} catch (error) {
console.error(`ErrorCode: ${(error as BusinessError).code},  Message: ${(error as BusinessError).message}`);
}
}

build() {
Column() {
Web({ src: 'www.example.com', controller: this.controller })
}
}
```

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/web/web-debugging-with-devtools.md`
**定位：** 1.1 无线调试 > 1.1.1 应用代码开启Web调试开关

## 代码 3: hdc工具调试命令示例

**功能说明：** 这些命令展示了如何使用hdc工具进行HAP包的安装、更新和卸载操作。hdc是HarmonyOS Device Connector的缩写，是一个命令行工具，用于与OpenHarmony设备进行通信和管理。这些命令支持批量操作，可以同时处理多个HAP文件，提高了开发效率。

**使用注意：** HAP的路径为开发平台上的文件路径，以Windows开发平台为例

**涉及API：** hdc install, hdc uninstall

```shell
# 安装、更新，多HAP可以指定多个文件路径
hdc install entry.hap feature.hap
# 执行结果
install bundle successfully.
# 卸载
hdc uninstall com.example.myapplication
# 执行结果
uninstall bundle successfully.
```

**来源文件：** `https://gitee.com/openharmony/docs/raw/master/zh-cn/application-dev/quick-start/hap-package.md`
**定位：** 1.5 调试

# 相关概念

- HAP（Harmony Ability Package）
- ArkTS编程语言
- API版本兼容性
- 设备连接调试
- WebView组件

# 推荐阅读

- **DevEco Studio使用指南**: 深入了解DevEco Studio的各项功能和高级用法
- **OpenHarmony应用开发入门**: 学习如何使用这些工具进行实际的OpenHarmony应用开发
- **hdc工具详细文档**: 掌握hdc命令行工具的所有功能和参数用法
