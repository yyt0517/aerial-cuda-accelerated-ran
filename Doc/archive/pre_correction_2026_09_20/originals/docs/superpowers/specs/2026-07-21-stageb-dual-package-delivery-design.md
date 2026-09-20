# Stage-B 双包交付设计

## 目标

在不改变已验证 Linux/CUDA 复现包的前提下，新增一个不含 Linux 外部符号链接的 Windows 源码浏览 ZIP，并用顶层说明文件明确三个交付文件的用途。

## 交付物

交付目录 `deliverables/` 包含：

1. `stageB_source_62227dc.zip`：Windows 源码浏览包；
2. `stageB_cuda_rerun_62227dc.tar.gz`：原始 Linux/CUDA 复现包，内容与现有归档保持不变；
3. `stageB_cuda_rerun_62227dc.tar.gz.sha256`：原始复现包完整性校验文件；
4. `README_FIRST.txt`：双包用途、Windows 解压提示和 Linux/CUDA 复现入口说明。

## Windows 浏览包范围

ZIP 的唯一顶层目录为 `stageB_source_62227dc/`。内容从已验证的 `deliverables/stageb_cuda_rerun_62227dc/` 生成，包含：

- C/C++、CUDA、Python、Shell 等源代码和脚本；
- CMake、JSON、YAML、INI、CSV 等项目配置或源码随附数据；
- Markdown、纯文本、许可证和报告文档；
- 文档引用的本地图片以及源码示例随附的必要资产。

明确排除：

- `output/` 下的参考运行结果；
- Linux 符号链接，包括指向 `/opt/nvidia/ldpc_decoder_cubin` 的外部链接；
- `SHA256SUMS`，因为它对应 Linux/CUDA 完整复现包而非过滤后的 ZIP；
- `.git`、构建目录、缓存、训练运行、检查点和导出物；
- 当前工作区未纳入原复现包的修改或未跟踪文件。

ZIP 不把符号链接转换为普通文件或占位文件，以避免让浏览者误认为外部 NVIDIA 运行组件属于项目源码。

## 原始包不变性

生成 Windows ZIP 前后分别校验原始 `tar.gz` 的 SHA256。其预期值保持为：

```text
1321254fe9b941536466f2c69033e59a1a764b1b3672b49f4f43b96cebded372
```

不重新生成、不修改原始归档和对应 `.sha256` 文件。

## README_FIRST.txt

说明文件采用用户给定的中文三项文件说明，并补充：

- Windows 浏览包不用于 CUDA 实验复现；
- Windows 解压原始包遇到符号链接权限提示时，仅浏览代码可全部跳过；
- 完整复现应在 Linux/CUDA 环境解压原始包并阅读 `REPRODUCE.md`。

## 生成方法

使用一个临时目录构造 `stageB_source_62227dc/`，从已验证复现目录复制普通文件并排除上述路径，随后用 ZIP 递归打包。生成过程不跟随符号链接，也不把临时目录写入仓库。

## 验证标准

- 原始复现包 SHA256 在操作前后均保持不变；
- `unzip -t stageB_source_62227dc.zip` 通过；
- ZIP 只有一个顶层目录 `stageB_source_62227dc/`；
- ZIP 中不存在符号链接、`output/`、`SHA256SUMS` 或私钥标记；源码和文档中正常出现的 Linux 路径文字不作为符号链接判据；
- ZIP 包含 `CMakeLists.txt`、`SOURCE_VERSION.txt`、`REPRODUCE.md`、Stage-B 报告、运行脚本、绘图脚本及测试；
- Windows 浏览包中的抽样源码文件与原复现目录逐字节一致；
- `README_FIRST.txt` 与实际文件名和用途一致。

## 非目标

- 不在 Windows 上构建或运行 CUDA 仿真；
- 不修改 Linux/CUDA 原始复现包；
- 不将参考仿真输出复制到 Windows 浏览包；
- 不提交或推送交付归档到远程仓库。
