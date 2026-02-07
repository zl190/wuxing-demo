# 五行计算 (Wuxing Compute)

> 用现代框架解构传统系统

用图神经网络的消息传递机制，形式化中国传统五行相生相克系统。

## 核心思路

```
时间周期 (52万) ←同构→ 八字符号 (52万) → 五行向量 (5)
                    ↑                   ↑
              换表示：引入语义+图         压缩
```

- **换表示**：八字编码引入五行语义 + 相生相克图结构（类似傅立叶变换的换基）
- **压缩**：22 符号 → 5 五行，权重聚合，图约束
- **消息传递**：相生加分，相克减分，模拟系统演化

详细解释见 [📖 博客文章](https://zl190.github.io/blog/wuxing-gnn)

## Demo

在线体验：[wuxing.ylabs.dev](https://wuxing.ylabs.dev)

两个版本：
- `/` — 用户优先版，快速体验
- `/ds` — DS Portfolio 版，展示建模思路

## 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 (用户版)
streamlit run app.py

# 运行 (DS版)
streamlit run app_v2.py
```

## 项目结构

```
├── app.py          # 用户优先版
├── app_v2.py       # DS Portfolio 版
├── prompts/        # LLM prompt 模板 (few-shot)
└── requirements.txt
```

## 技术栈

- Streamlit (UI)
- NumPy (矩阵运算)
- Plotly (可视化)
- OpenAI API (可选，AI 解读)

## 这不是什么

- ❌ 不是算命工具
- ❌ 不是可验证的因果模型
- ✅ 是符号系统的形式化建模
- ✅ 是"用现代框架解构传统系统"的案例

## License

MIT
