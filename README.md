# TikZ Native Provider

这是 `tex-to-mathcapture-ppt` 仓库中受 Git 管理的 TikZ → Manim 原生转换器。
编辑器仍然只通过版本化 JSON Bridge 调用它，不直接导入转换器实现。

## 目录

- `tikz_native/`：Provider 源码、schema 与 TikZ 子集定义；
- `tests/`：二维、三维、Bridge、兼容性和 Provider 回归；
- `examples/`：使用本仓库便携 TikZ 夹具的原生场景示例；
- `scripts/`：转换、兼容性审计和基线验证工具；
- `docs/`：转换规则、受控子集、三维原型和接入说明。

本目录不建立嵌套 `.git`；所有文件与 PPT 编辑器共用仓库根目录的 Git 历史。
版本标签使用 `tikz-native-provider-v<version>` 命名，避免与整套编辑器的发布标签混淆。

## 建立环境

```bash
cd /path/to/tex-to-mathcapture-ppt/tools/tikz-native-provider
python3 -m venv .venv-manim
.venv-manim/bin/python -m pip install -r requirements.txt
```

## 健康检查

```bash
.venv-manim/bin/python -m tikz_native.bridge health
```

也可以使用另一套已经安装 Manim 0.20.1 的 Python；调用端会把本目录放入
`PYTHONPATH`，因此虚拟环境不需要进入 Git。

## 回归测试

```bash
.venv-manim/bin/python -m unittest discover -s tests -p 'test_tikz_native_*.py'
```

修改 Provider 后必须确认 health 返回新的 `provider.revision`，并在 PPT 工作副本中
显式重新转换受影响的 TikZ 资产。禁止用新版 Provider 静默重建旧资产。

回归所需的全国一卷 TikZ 已提取为只包含颜色、公共绘图宏和 16 个 `tikzpicture` 的
便携夹具；测试不再依赖 `/Users/.../讲评课` 中的个人讲义文件。

旧的 2026-08-03 `tar.gz` 仍作为迁入前快照保留；当前完整历史应以主仓库提交、
`tikz-native-provider-v*` 标签和仓库外 Git bundle 为准。
