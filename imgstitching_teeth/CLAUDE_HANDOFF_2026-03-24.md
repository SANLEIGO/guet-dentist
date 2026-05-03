# Claude Handoff - 2026-03-24

## 当前结论

项目当前已回退到用户认可的 teeth-only 稳定版本：

- 当前分支：`main`
- 当前提交：`37fb5e5` `实现牙齿主体拼接输出`

这次回退是用户明确要求执行的，原因是后续几轮“去噪 / 主连通域 / 后处理”尝试都会让 **4 张图拼接结果出现严重畸变**，而最早的 `stitched_teeth_only_v1.png` 版本虽然仍有少量残留噪声，但整体结构是正常的。

## 当前 git 状态

未跟踪文件：

- `diagnostics_teeth_only_v1.json`
- `stitched_teeth_only_v1 (1).png`
- `stitched_teeth_only_v1 (2).png`
- `stitched_teeth_only_v1 (3).png`

这些文件是用户提供/生成的调试产物，暂未删除。

## 用户当前真实诉求

用户要的不是整张口腔照片拼接，而是：

- 输入：包含牙齿、舌头、软组织等的口腔照片
- 输出：**只包含牙齿主体** 的拼接图

并且用户还有以下稳定要求：

1. 一次只上传一个区域的相关图片
   - 前端按 `左上 / 左下 / 右上 / 右下` 引导
2. 低质量图片也要尽量参与拼接
3. 支持手工排序，且手工排序是推荐路径
4. 分阶段改，每一步都要能看出效果
5. 不要无端“越改越复杂”，要优先保持当前可工作的版本

## 当前版本已经具备的能力

基于 `37fb5e5` 这个版本：

- Streamlit 前端支持多图上传
- 前端支持区域选择
- 前端支持分割预览
- 前端支持手工选择/排序拼接顺序
- 分割使用 AlphaDent + GrabCut
- 拼接输出语义已经切换成 teeth-only
- 质量门控与 diagnostics JSON 已接入前端
- 两图拼接基本正常
- 四图在“原始 teeth-only 版本”下可以得到整体可接受结果

## 已确认失败的方向

后续如果继续工作，**优先不要立刻重复下面这些尝试**，因为已经验证过会出问题：

### 1. 对中间累计结果做 aggressive 去噪 / 形态学清理

曾尝试过：

- `_clean_stitched_mask()`
- 角落伪影去除
- 主连通域筛选
- 根据中心性/面积保留单一 component

结果：

- 2 张图时有时看起来改善了噪声
- 但 4 张图时会引发明显畸变、扇形拉伸、结构错乱

### 2. 直接从 stitched RGB 非零像素反推 mask，再做区域筛选

曾做过类似逻辑：

- `_extract_mask_from_image(outputs.stitched)`
- 然后再 `_select_main_dental_region(...)`

推断问题：

- warp/blend 后的插值残留、细线、非零伪影会被当作真实 mask
- 后续连通域选择会把错误区域当成主体，导致最终输出严重畸变

注意：虽然这个方向在理论上像是 root cause 之一，但**用户已经要求整体回退重来**，因此不要直接沿着上一轮未验证成功的 patch 继续堆代码。

## 本次回退前最后一次思考结论

在回退前，最后一次分析认为：

- `blend_images()` 已经会返回 `stitched_mask`
- 真正危险的地方是 `pipeline.py` 中后续步骤把这个 mask 丢掉了
- 多图累计和最终清理都存在“从 RGB 图反推 mask”的问题

但：

- 用户已明确表示“还是不行”
- 并要求回退到上一个 git 版本重新开始

因此现在**不要假设这个分析已经被证实**，它只是一个待验证思路，而不是当前事实结论。

## 重新开始时的正确策略

建议下一次从 `37fb5e5` 这个稳定点重新做，而且遵守下面顺序：

### 第一步：先只做代码阅读和对比，不急着改

重点重读：

- `dental_stitcher_v1/pipeline.py`
- `dental_stitcher_v1/blending.py`
- `dental_stitcher_v1/segmentation.py`
- `app.py`

要回答的问题：

1. 当前稳定版本下，为什么 `stitched_teeth_only_v1.png` 虽然有噪声但结构没坏？
2. 4 图模式相对 2 图模式，累计参考图是如何演化的？
3. 当前真正该被约束的是：
   - 输入 mask？
   - blending canvas？
   - 累计参考语义？
   - 还是最后一步输出裁剪？

### 第二步：只做“最小改动”的单点实验

不要一次做多件事。

推荐顺序：

1. 先只增加调试信息，不改算法语义
2. 再只改一个点
3. 跑用户固定样本
4. 停下来观察结果

### 第三步：优先保证 4 图不畸变，再谈去噪

用户已经明确给出信息：

- 没动去噪前，4 图是“基本正常”的
- 去噪之后，4 图“严重畸变”

所以接下来工作的优先级是：

1. 先守住结构稳定
2. 再处理噪声
3. 不要为了去噪破坏整体几何

## 关键文件提示

### `app.py`

当前前端已经是 teeth-only 语义，主要包括：

- 区域选择
- 手工排序
- 分割预览
- 输出下载 `stitched_teeth_only_v1.png`
- diagnostics 下载

### `dental_stitcher_v1/pipeline.py`

这是下一轮最关键的文件。

重点关注：

- `run_pipeline(...)`
- `_run_pair_pipeline(...)`
- `_run_multi_pipeline(...)`
- `_segmentation_from_mask(...)`
- `_extract_teeth_image(...)`

如果后续重新分析 4 图畸变，核心仍然会在这里。

### `dental_stitcher_v1/blending.py`

已经是 mask-aware blending，不是最先该怀疑的对象，但需要确认：

- 返回的 mask 与 image 的语义是否一致
- 多步累计后 occupancy / crop 是否带来意外几何影响

## 最近提交历史

```text
37fb5e5 实现牙齿主体拼接输出
f69365f 修复 datetime 导入冲突错误
cbc45bf 完善前端分割交互逻辑，提供完整的用户体验
2a1a720 简化项目：删除旧版拼接器和龋齿ROI逻辑，统一使用AlphaDent+GrabCut分割
46dc4f0 Add AlphaDent GrabCut refinement for segmentation
```

## 给下次 Claude 的直接启动建议

下次进入项目后，直接做这几步：

1. 先确认当前仍在提交 `37fb5e5`
2. 读取：
   - `dental_stitcher_v1/pipeline.py`
   - `dental_stitcher_v1/blending.py`
   - `app.py`
   - `diagnostics_teeth_only_v1.json`
3. 不要先写代码，先总结：
   - 稳定版本为何能工作
   - 之前几次后处理为何会破坏 4 图
4. 然后只提出一个最小实验改动
5. 改一点，停一下，等用户验证

## 一句话交接

**现在的正确起点不是“继续修补上次失败的后处理”，而是“以 `37fb5e5` 这个稳定 teeth-only 版本为基线，重新分析 4 图累计语义，做最小、可验证的单点改动”。**
