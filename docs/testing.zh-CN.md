# 测试适用范围

[首页](../README.zh-CN.md) · [运维与开发](operations.zh-CN.md) · [English](testing.md) | 中文

Carrel 将源码回归、部署检查和真实知识库评测分别维护。每一层的结果对应其实际覆盖的行为。

| 层次 | 代表测试 | 验证内容 |
| --- | --- | --- |
| 解析与原文保真 | `test_source_fidelity`、`test_pdf_evidence`、`test_short_table_recovery`、`test_section_provenance`、`test_visual_retention` | 使用合成输入和模型桩，验证解析、切块、证据回读保留原文及定位。 |
| 图谱与本体 | `test_graph_integrity`、`test_graph_semantics`、`test_ontology_invariants`、`test_temporal_schema_reuse`、`test_knowledge_foundation` | 实体身份、谓词约束、来源归属、时间范围和图谱版本行为。 |
| 治理 | `test_correction_api`、`test_enrollment_corrections` | 更正校验和适用范围、证据失效、知识库身份隔离及并发修改。 |
| 检索与智能体 | `test_search`、`test_retrieval_quality`、`test_dual_rerank`、`test_search_skill` | 排名、证据质量、来源范围与智能体接口结构。 |
| 生命周期与控制台 | `test_scanner_worker`、`test_graph_build`、`test_console_api`、`test_console_stage_language` | 队列归属、取消、续跑、版本发布、中英文显示和进度。 |
| 公共部署 | `test_deployment`、`test_remote_vlm_deployment`、`test_reranker_endpoints`、`test_maintenance_ops` | 公共 Compose 组合、接口配置、本地服务控制及维护模板。 |
| 平台集成 | `test_macos_sync` 及 `test_maintenance_ops` 中的已安装单元检查 | 特定主机的脚本行为和服务定义。 |

表中名称对应 `app/tests/` 下的 `.py` 文件。在 `app/` 中运行 `.venv/bin/python -m pytest -q`。JavaScript 翻译测试还需要 `PATH` 中可用的 Node.js，CI 显式安装 Node.js。平台相关跳过表示本轮未执行该主机专属行为。

## 与私有部署对照

通用行为断言保持一致：解析内容、来源身份、排名、更正范围、显式时间筛选、取消和图谱发布，在私有部署与公共包中的语义相同。

部署断言按照各自安装方式核对：

- 公共包使用 `carrel-*` 容器和 systemd 单元、可配置接口及 loopback 默认值。私有部署可使用不同名称、路径、监听地址和模型设置。
- 公共 MinerU 容器按主机选择 CPU/GPU 后端，私有部署可固定后端；分别与各自的 Compose 和服务配置核对。
- 后端供人阅读的消息可能使用不同语言。先断言稳定字段、状态变化和错误条件，再用实际翻译代码验证中英文界面。文件名、实体名保留原文。
- 公共套件覆盖托管模型接口的兼容性。私有接口接受哪些参数，在该服务部署或调整时单独验证。
- 按测试目的和行为建立对应关系。一份私有审计文件可对应多个公共功能测试文件；重命名仍保留原用例和负例断言。

安装路径、凭据、私有题集和原文件放在公共源码目录之外。公共夹具使用合成知识库、文档、实体和接口地址。

## 真实服务评测

`search eval`、原文到切块核对、图谱事实检查和全新安装冒烟验证，与使用服务桩的回归套件分别记录。结果同时记录代码版本、模型 ID、向量维度、启用通道和题集。更换模型后，原检索分数仍代表原模型配置，详见[检索评测](retrieval.zh-CN.md#检索评测)。
