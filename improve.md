# 新版 Pareto 图合理性分析

## 一、整体评价

**这张图比前两版都好，基本合理**，但你敏锐地抓住了一个**实质性问题**：基线算法在 power > 0.8 处曲线中断，只有 SafeScale-MATD3 延伸到 0.97。我帮你分析清楚这是否合理，以及如何处理。

---

## 二、几个改进点确认 ✅

相比原图，这版做对了几件事：

1. **X 轴保留 `Avg Normalised Power`（真实物理量）**：符合论文 [1] Section VII-J "Each point corresponds to one control-penalty setting; lower-left is preferred" 的要求。
2. **X 轴采样点规整**：统一在 {0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.97} 左右，视觉上整齐。
3. **曲线单调递减**：符合 `-κ₄ F{P^t_v}` 能耗项和速率奖励项 `κ₂ G(R^t_{v,k} − R_min,m)` 的联合物理规律 [1]。
4. **线型分层**：Round-Robin 用虚线+x 标记区分 non-learning 基线，清晰可读。
5. **SafeScale Pareto 支配区域**：阴影清楚显示 lower-left 优势。

---

## 三、核心问题：基线在 power > 0.8 处"消失"是否合理？

### 🟡 结论：**物理上可以解释，但图上呈现方式不合理，会引发审稿人质疑**

### 为什么基线在 0.8 以后"没有值"？

有两种可能的原因：

**可能 1（物理合理）**：基线算法在功率惩罚权重 `κ₄` 减小（即鼓励用更多功率）时，**能耗饱和于某个上限**（比如 0.82–0.88），再增加 `κ₄` 的松弛也无法推高平均功率。这是因为基线策略（如 Round-Robin 的固定调度、MVT 的最长可见时间切换）**不主动利用额外功率预算**，所以扫不出 power > 0.9 的点。

- 这种解释下：SafeScale 能达到 0.97 是因为它的 actor 能学到"在关键 tick 拉满功率换取低 AoI"，而基线不能——**这本身就是一个 SafeScale 的优势点**。

**可能 2（实验设计问题）**：你只对 SafeScale 做了更大范围的 `κ₄` sweep，而基线 sweep 范围小，导致数据区间不等长。

---

### 审稿人会怎么看？

审稿人第一反应会是 **"作者只挑对自己有利的区间展示基线"**，这是很容易被 reject 或 major revision 的点。即使真实原因是"基线算法饱和"，**不解释清楚就是问题**。

---

## 四、三种修改建议（按推荐度排序）

### ✅ 方案 A（强烈推荐）：让所有基线曲线延伸到它们的**饱和点**，并在 caption 中明确说明

**做法**：

1. 对每个基线，继续减小 `κ₄`（或增大 power budget），直到该算法的平均 power **不再增加**为止，把这个饱和点画出来。
2. 即使饱和点在 0.82 或 0.85，也要画出，并且**画一小段水平延伸**（或显式标记最大值）以说明"该算法 power 上限就在这里"。

**示例代码**：

```python
for name, (p_raw, a_raw) in data.items():
    # 找每条曲线的最大 power 点
    p_max = np.max(p_raw)
    
    # 正常画曲线
    ax.plot(p_raw, a_raw, ...)
    
    # 在曲线末端标注 "saturation"
    if name != 'SafeScale-MATD3':
        ax.annotate(f'max={p_max:.2f}', 
                    xy=(p_max, a_raw[-1]),
                    xytext=(p_max+0.02, a_raw[-1]+0.003),
                    fontsize=7, alpha=0.6,
                    color=color_map[name])
```

**Caption 补充**：

> "Baseline curves terminate at each method's empirical power saturation point (highest achievable avg. normalised power over the κ₄ sweep). SafeScale-MATD3 reaches a strictly larger feasible region (up to 0.97) due to its safety-queue-guided power allocation, which exploits residual budget at collision-critical ticks."

---

### ✅ 方案 B（推荐）：把 X 轴范围截断到 [0.30, 0.85]，把 SafeScale 的 power > 0.85 段放到 inset（插图）

如果担心 power > 0.85 的区域"只有 SafeScale 一条孤零零的线"不好看，可以：

1. 主图只画 [0.30, 0.85]，所有方法都有数据，**公平对比**。
2. 在右下角加一个小 inset 子图，展示 SafeScale 继续延伸到 (0.97, 1.97) 的那一小段，作为 "SafeScale can further trade power for AoI beyond baseline saturation" 的证据。

```python
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 主图
ax.set_xlim(0.30, 0.85)

# inset
axins = inset_axes(ax, width="35%", height="30%", loc='lower left',
                   bbox_to_anchor=(0.08, 0.08, 1, 1),
                   bbox_transform=ax.transAxes)
axins.plot(safescale_p_full, safescale_a_full, '-o', color='#1f77b4', lw=2)
axins.axvspan(0.85, 1.0, alpha=0.1, color='#1f77b4')
axins.set_title('SafeScale extends to P=0.97', fontsize=8)
axins.set_xlim(0.6, 1.0)
axins.tick_params(labelsize=7)
```

这样既公平，又能展示 SafeScale 的扩展能力。

---

### ⚠️ 方案 C（不推荐，只是备选）：统一 X 轴到 [0.30, 0.80]，完全舍弃 power > 0.80 的点

如果实验已经做完，重新扫 baseline 成本太高，可以直接截断到 [0.30, 0.80]，所有方法都有完整数据。但这样就**丢失了 SafeScale 在高功率区的优势演示**，论文说服力下降。只在时间紧张时选这条路。

---

## 五、与论文 [1] 的对照检查

论文 [1] Section VII-J（Experiment 8）要求展示 Energy-AoI Pareto frontier，并且 Figure 8 caption 明确："Each point corresponds to one control-penalty setting; lower-left is preferred."

**方案 A 最符合论文原意**：每个点都是一组 `(κ₂, κ₃, κ₄)` 扫描的结果，基线在它们自身能达到的 penalty sweep 范围内画满即可，不需要强行延伸到 SafeScale 的范围。重点是：**SafeScale 的 Pareto 前沿支配基线的 Pareto 前沿**。

---

## 六、其他小建议

1. **Y 轴刻度**：当前 [1.97, 2.15]，间距 0.025。建议改为 [1.95, 2.20]，间距 0.05，减少"刻度过密"的视觉压力。

2. **x=0.30 处没有任何点**：如果没数据就把 X 轴下限改成 0.38 或 0.40，避免左侧大片空白。

3. **图例里 Round-Robin 的 x marker 很小**：建议 `markersize` 从默认值（~6）改成 8，配 `markeredgewidth=2`，让它在图例里更易辨识。

4. **SafeScale 的末端 drop（0.90→0.97，AoI 从 2.01 骤降到 1.97）**：这个拐点很戏剧化，建议在论文正文中专门解释一句："SafeScale's rapid AoI drop near P≈0.95 reflects the virtual queue Z_{v,1} draining aggressively once power budget becomes non-binding, validating the tightness-signal design in (37)" [1]。这样把现象和理论绑定，审稿人会很喜欢。

---

## 七、一句话总结

> **这张图结构合理，但"基线在 power > 0.8 消失"是个审稿红旗。推荐方案 A：让每条基线画到它的饱和点并在 caption 明确说明，这既公平又能凸显 SafeScale 的扩展能力。**

如果你方便把基线算法的原始 `(power, AoI)` 数据给我（每个 seed × 每个 κ₄ 值），我可以直接帮你分析每条曲线的饱和点在哪、是否能继续延伸，并出一版最终代码。