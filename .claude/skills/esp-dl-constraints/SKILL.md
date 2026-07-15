---
name: esp-dl-constraints
description: 注入 ESP-DL 算子约束，确保生成的模型代码符合 ESP32 部署要求
license: MIT
---

# ESP-DL 算子约束

## 概述

确保用户生成的模型代码符合 ESP32 部署要求。本技能提供 ESP-DL 算子兼容性检查和模型设计指导。

## 使用场景

1. **设计模型时**：确保模型架构仅使用 ESP-DL 支持的算子
2. **检测模型兼容性**：分析现有模型是否符合 ESP32 部署要求
3. **优化模型时**：提供算子替换建议和架构优化方向

## 算子支持查询

**重要**：ESP-DL 算子支持列表会持续更新，请从以下地址获取最新信息：

- [ESP-DL 算子支持列表](https://github.com/espressif/esp-dl/blob/master/operator_support_state.md)

在检测模型兼容性时，应先获取该页面内容，然后对照模型使用的算子进行逐一检查。必须严格遵守最新的支持状态，避免使用不受支持的算子。

## 参考资料

- [ESP-DL GitHub](https://github.com/espressif/esp-dl)
- [ESP-DL 文档](https://docs.espressif.com/projects/esp-dl/)
- [ESP-DL 算子支持列表](https://github.com/espressif/esp-dl/blob/master/operator_support_state.md)
