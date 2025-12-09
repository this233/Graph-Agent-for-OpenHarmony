# 用户问题
What is OpenHarmony's architecture?

# 核心回答
OpenHarmony is a distributed, micro-kernal-based operating system designed for all-scenario intelligent devices. It adopts a layered architecture consisting of Kernel Layer, System Service Layer, Framework Layer and Application Layer. The system service layer is further divided into core and optional services to enhance modularity and flexibility. OpenHarmony emphasizes distributed capability, security, and hardware abstraction to enable seamless cross-device collaboration and efficient resource utilization.

## 关键要点
- Layered architecture: Kernel, System Service, Framework, Application
- Micro-kernel design for high security and reliability
- Distributed Virtual Bus enables seamless device interconnection
- Hardware Driver Foundation (HDF) provides unified hardware abstraction
- Modular system services split into core and optional components

# 详细内容

## 1. Overall Architecture Overview

OpenHarmony’s architecture is divided into four layers from bottom to top: the Kernel Layer, System Service Layer, Framework Layer, and Application Layer. The Kernel Layer supports both Linux kernel and LiteOS, with the latter optimized for lightweight IoT devices. The System Service Layer is split into core system services (mandatory for minimal system boot) and optional system services (loaded on demand). The Framework Layer exposes APIs to applications, while the Application Layer hosts user apps. This layered design ensures scalability across devices ranging from 128 KB to GBs of memory.

**本节要点：**
- Four-layer stack ensures clear separation of concerns
- Kernel abstraction allows multiple kernel backends
- System services are modularized for footprint optimization

**来源：**
- 文件: `OpenHarmony_Architecture_Overview.md`
  定位: section 1
  相关性: Describes the four-layer model and kernel choices

## 2. Kernel Layer and LiteOS

The Kernel Layer supports two kernels: the standard Linux kernel for rich devices and LiteOS for lightweight devices. LiteOS is a real-time micro-kernel offering fast interrupt response, low power consumption, and a minimum RAM footprint of 128 KB. It provides thread scheduling, IPC, memory management, and timer services. LiteOS adopts MPU instead of MMU to reduce hardware requirements. For security, the kernel isolates user and kernel space and supports capability-based access control. Drivers are abstracted through the Hardware Driver Foundation (HDF) so that upper layers remain kernel-agnostic.

**本节要点：**
- LiteOS targets MCU-class devices with < 128 KB RAM
- MPU-based memory protection lowers silicon cost
- HDF unifies driver models across Linux and LiteOS

**来源：**
- 文件: `LiteOS_Design_WhitePaper.md`
  定位: chapter 3
  相关性: Details LiteOS micro-kernel features and memory footprint

## 3. System Service Layer

The System Service Layer contains all system functionalities divided into core and optional services. Core services include distributed scheduling, soft bus, security, and basic graphics, guaranteeing minimal boot and inter-device connectivity. Optional services such as telephony, location, and multimedia are installed dynamically according to device capability. Each service runs in an independent sandbox to achieve fault isolation. Inter-process communication uses an object-oriented RPC mechanism optimized for low latency. Service management supports lifecycle callbacks for installation, startup, and recovery, enabling hot-plug behavior across the distributed network.

**本节要点：**
- Core services guarantee minimal boot capability
- Optional services reduce memory footprint on constrained devices
- Sandbox isolation enhances system reliability

**来源：**
- 文件: `System_Service_Layer_Design.md`
  定位: section 2.1-2.3
  相关性: Explains core vs optional services and sandboxing

## 4. Distributed Soft Bus

Distributed Soft Bus is a key innovation providing auto-discovery, authentication, and high-speed data transmission among OpenHarmony devices. It abstracts network details, allowing developers to invoke remote capabilities as if they were local. The bus supports Wi-Fi, BLE, and Ethernet with adaptive channel selection and QoS guarantee. Security is enforced through device certificates and session keys negotiated during authentication. The API surface is unified across languages (JS, C/C++, ArkTS), enabling write-once-run-everywhere distributed applications. Performance benchmarks show < 5 ms discovery latency and > 200 Mbps throughput on Wi-Fi 5 GHz.

**本节要点：**
- Transparent remote invocation simplifies distributed programming
- Multi-link aggregation boosts throughput
- Security based on PKI and per-session keys

**来源：**
- 文件: `Distributed_SoftBus_Spec.md`
  定位: chapter 4
  相关性: Details protocol, performance, and security of soft bus

## 5. Framework and Application Layer

The Framework Layer exposes two programming models: Stage (for FA feature ability) and FA (for PA particle ability). It includes UI, notification, notification, distributed data management, and background task APIs. ArkUI, a declarative UI framework, renders native components with GPU acceleration. The Application Layer packages apps in HAP (Harmony Ability Package) format containing code, resources, and config. Each HAP is signed and sandboxed. Apps can be deployed atomically or in bundles, supporting modular upgrade. The ability lifecycle is managed by the system scheduler, enabling migration across devices without user intervention.

**本节要点：**
- ArkUI offers reactive programming model
- HAP format supports modular deployment
- Ability migration enables seamless user experience

**来源：**
- 文件: `App_Framework_Guide.md`
  定位: section 5
  相关性: Explains ArkUI and HAP structure

# 相关图片

## 图片 1
![Four-layer OpenHarmony architecture diagram showing Kernel, System Service, Framework, and Application layers.](https://example.com/docs/openharmony_arch.png)

**说明：** Four-layer OpenHarmony architecture diagram showing Kernel, System Service, Framework, and Application layers.

**上下文：** Presented in the architecture overview chapter to visualize layer separation and key components.

**与问题的相关性：** Helps readers quickly grasp the layered design and major subsystems.

**尺寸：** 1200x800
**建议显示方式：** full-width

**来源文件：** `OpenHarmony_Architecture_Overview.md`
**定位：** figure 1

# 相关表格

## 表格 1: Kernel Comparison

**描述：** Feature comparison between Linux kernel and LiteOS used in OpenHarmony Kernel Layer.

| Feature        | Linux Kernel | LiteOS |
|----------------|--------------|--------|
| Memory Footprint | ~2 MB       | 128 KB |
| MMU/MPU        | MMU          | MPU    |
| Boot Time      | < 1 s        | < 20 ms|
| Target Device  | Rich device  | MCU    |
| Real-time      | Soft RT      | Hard RT|

**关键数据：**
- LiteOS minimum footprint 128 KB
- LiteOS boot time < 20 ms
- Linux supports full MMU while LiteOS uses MPU

**来源文件：** `LiteOS_Design_WhitePaper.md`
**定位：** table 2-1

# 相关代码

## 代码 1: HDF Driver Registration Example

**功能说明：** This snippet shows how to register a sample driver with the Hardware Driver Foundation. HDF_INIT macro places the driver entry into a special section so the HDF manager can enumerate and initialize it automatically during boot.

**使用注意：** Ensure moduleName is unique across system; return HDF_SUCCESS to indicate successful initialization.

**涉及API：** HDF_INIT, HdfDeviceObject, HdfDriverEntry

```c
static int32_t HdfSampleDriverInit(struct HdfDeviceObject *deviceObject) {
    if (deviceObject == NULL) {
        HDF_LOGE("Sample driver init failed!");
        return HDF_FAILURE;
    }
    HDF_LOGD("Sample driver init success");
    return HDF_SUCCESS;
}

struct HdfDriverEntry g_sampleDriverEntry = {
    .moduleVersion = 1,
    .moduleName = "sample_driver",
    .Init = HdfSampleDriverInit,
};

HDF_INIT(g_sampleDriverEntry);
```

**来源文件：** `HDF_Driver_Development.md`
**定位：** listing 3-2

# 相关概念

- Micro-kernel architecture
- Distributed operating system
- Hardware Abstraction Layer (HAL)
- Ability framework
- Inter-Process Communication (IPC)

# 推荐阅读

- **OpenHarmony Distributed Scheduling White Paper**: Explains how tasks are migrated across devices in detail.
- **ArkUI Declarative Programming Guide**: Deep dive into UI framework used in OpenHarmony.
