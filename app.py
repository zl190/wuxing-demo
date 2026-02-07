# -*- coding: utf-8 -*-
"""
计算人文 - 五行可视化与解读
Computational Humanities - Five Elements Visualization

Run: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date
from pathlib import Path
from openai import OpenAI

# ============ Constants ============
ELEMENTS = ['木 Wood', '火 Fire', '土 Earth', '金 Metal', '水 Water']
ELEMENTS_SHORT = ['木', '火', '土', '金', '水']
WOOD, FIRE, EARTH, METAL, WATER = range(5)

# Heavenly Stems (天干) -> Element mapping
STEMS = {
    '甲': WOOD, '乙': WOOD,
    '丙': FIRE, '丁': FIRE,
    '戊': EARTH, '己': EARTH,
    '庚': METAL, '辛': METAL,
    '壬': WATER, '癸': WATER
}

# Earthly Branches (地支) -> Element mapping (simplified, main element only)
BRANCHES = {
    '子': WATER, '丑': EARTH, '寅': WOOD, '卯': WOOD,
    '辰': EARTH, '巳': FIRE, '午': FIRE, '未': EARTH,
    '申': METAL, '酉': METAL, '戌': EARTH, '亥': WATER
}

STEM_LIST = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
BRANCH_LIST = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']

# ============ Graph Matrices ============
def build_generating_matrix():
    """A_gen[i,j] = 1 means i generates j"""
    A = np.zeros((5, 5))
    A[WOOD, FIRE] = 1    # 木生火
    A[FIRE, EARTH] = 1   # 火生土
    A[EARTH, METAL] = 1  # 土生金
    A[METAL, WATER] = 1  # 金生水
    A[WATER, WOOD] = 1   # 水生木
    return A

def build_controlling_matrix():
    """A_ctl[i,j] = 1 means i controls/overcomes j"""
    A = np.zeros((5, 5))
    A[WOOD, EARTH] = 1   # 木克土
    A[EARTH, WATER] = 1  # 土克水
    A[WATER, FIRE] = 1   # 水克火
    A[FIRE, METAL] = 1   # 火克金
    A[METAL, WOOD] = 1   # 金克木
    return A

A_GEN = build_generating_matrix()
A_CTL = build_controlling_matrix()

# ============ BaZi Calculation (Simplified) ============
def get_stem_branch(year, month, day, hour):
    """
    Simplified BaZi calculation.
    Note: Real BaZi requires lunar calendar and solar terms. This is approximate.
    """
    # Year pillar (approximate)
    year_stem_idx = (year - 4) % 10
    year_branch_idx = (year - 4) % 12

    # Month pillar (very simplified)
    month_stem_idx = ((year - 4) % 5 * 2 + month) % 10
    month_branch_idx = (month + 1) % 12

    # Day pillar (simplified using a base date)
    base = datetime(1900, 1, 31)  # Known: 甲子日
    target = datetime(year, month, day)
    days_diff = (target - base).days
    day_stem_idx = days_diff % 10
    day_branch_idx = days_diff % 12

    # Hour pillar
    hour_branch_idx = ((hour + 1) // 2) % 12
    hour_stem_idx = (day_stem_idx % 5 * 2 + hour_branch_idx) % 10

    return [
        (STEM_LIST[year_stem_idx], BRANCH_LIST[year_branch_idx]),
        (STEM_LIST[month_stem_idx], BRANCH_LIST[month_branch_idx]),
        (STEM_LIST[day_stem_idx], BRANCH_LIST[day_branch_idx]),
        (STEM_LIST[hour_stem_idx], BRANCH_LIST[hour_branch_idx])
    ]

def bazi_to_matrix(pillars):
    """Convert BaZi pillars to 5x4 element distribution matrix"""
    X = np.zeros((5, 4))
    for col, (stem, branch) in enumerate(pillars):
        X[STEMS[stem], col] += 1
        X[BRANCHES[branch], col] += 1
    return X

# ============ LLM Narrative ============
def load_prompt_template():
    """Load prompt template from file"""
    prompt_file = Path(__file__).parent / "prompts" / "bazi_analyst.md"
    return prompt_file.read_text(encoding="utf-8")

def build_prompt(template, pillars, s, day_status):
    """Build the final prompt by replacing placeholders"""
    element_names = ['木', '火', '土', '金', '水']
    dist_parts = [f"{element_names[i]}={s[i]:.1f}" for i in range(5)]

    return template.format(
        year_pillar=f"{pillars[0][0]}{pillars[0][1]}",
        month_pillar=f"{pillars[1][0]}{pillars[1][1]}",
        day_pillar=f"{pillars[2][0]}{pillars[2][1]}",
        hour_pillar=f"{pillars[3][0]}{pillars[3][1]}",
        elements_dist=', '.join(dist_parts),
        day_status=day_status
    )

def generate_llm_narrative(api_key, prompt, model="gpt-4o-mini"):
    """Generate narrative interpretation using LLM"""
    try:
        # Validate API key format
        api_key = api_key.strip()
        if not api_key.isascii():
            return "❌ API Key 格式错误：包含非 ASCII 字符。OpenAI key 应该是 sk-... 格式，只含英文字母和数字。"
        if not api_key.startswith("sk-"):
            return "❌ API Key 格式错误：应该以 sk- 开头。"

        client = OpenAI(api_key=api_key)

        # o-series and gpt-5 models use max_completion_tokens, no temperature
        if model.startswith("o") or model.startswith("gpt-5"):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4000
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )
        return response.choices[0].message.content
    except Exception as e:
        import traceback
        return f"LLM 调用失败: {str(e)}\n\n详细信息:\n{traceback.format_exc()}"

# ============ Dynamics ============
def message_passing_step(h, lam_g=0.5, lam_c=0.5, rho=0.5):
    """One step of damped message passing with ReLU"""
    msg = h + lam_g * (A_GEN.T @ h) - lam_c * (A_CTL.T @ h)
    h_next = (1 - rho) * h + rho * msg
    return np.maximum(h_next, 0)

def simulate(h0, T=30, **kwargs):
    """Run simulation for T steps"""
    trajectory = [h0.copy()]
    h = h0.copy()
    for _ in range(T):
        h = message_passing_step(h, **kwargs)
        trajectory.append(h.copy())
    return np.array(trajectory)

# ============ Visualization ============
def plot_radar(s, title="五行分布"):
    """Radar chart for element distribution"""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(s) + [s[0]],
        theta=ELEMENTS + [ELEMENTS[0]],
        fill='toself',
        name='五行强度'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(s)*1.2+1])),
        showlegend=False,
        title=title,
        height=350
    )
    return fig

def plot_trajectory(trajectories, element_idx=EARTH, element_name="土 Earth", intervene_name="火 Fire"):
    """Plot element value over iterations for different interventions"""
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (delta, traj) in enumerate(trajectories.items()):
        fig.add_trace(go.Scatter(
            x=list(range(len(traj))),
            y=traj[:, element_idx],
            mode='lines',
            name=f'δ = {delta}',
            line=dict(width=2, color=colors[i % len(colors)])
        ))
    fig.update_layout(
        xaxis_title="迭代 Iteration",
        yaxis_title=f"{element_name} 强度",
        title=f"干预实验: do({intervene_name.split()[0]} += δ) → {element_name} 变化",
        height=400
    )
    return fig

def plot_graph():
    """Plot the Wuxing graph structure"""
    # Pentagon coordinates
    angles = [np.pi/2 - 2*np.pi*i/5 for i in range(5)]
    x = [np.cos(a) for a in angles]
    y = [np.sin(a) for a in angles]

    fig = go.Figure()

    # Generating edges (adjacent, green)
    for i in range(5):
        j = (i + 1) % 5
        fig.add_trace(go.Scatter(
            x=[x[i], x[j]], y=[y[i], y[j]],
            mode='lines',
            line=dict(color='green', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Controlling edges (skip one, red, dashed)
    for i in range(5):
        j = (i + 2) % 5
        fig.add_trace(go.Scatter(
            x=[x[i], x[j]], y=[y[i], y[j]],
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers+text',
        marker=dict(size=40, color=['green', 'red', 'brown', 'gold', 'blue']),
        text=ELEMENTS_SHORT,
        textposition='middle center',
        textfont=dict(size=16, color='white'),
        hovertext=ELEMENTS,
        hoverinfo='text'
    ))

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x'),
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        title="五行关系图 (绿=相生, 红=相克)"
    )
    return fig

# ============ Streamlit App ============
st.set_page_config(page_title="计算人文", layout="wide")

# Query param for tab selection
query_tab = st.query_params.get("tab", "体验")
default_tab_idx = 1 if query_tab == "lab" else 0

st.title("计算人文")
st.markdown("### 五行")
st.caption("用现代框架解构传统系统")

# ============ Sidebar: Shared Input ============
st.sidebar.header("📅 输入")

input_mode = st.sidebar.radio("输入方式", ["生日推算(简化)", "手动八字"])

if input_mode == "生日推算(简化)":
    birth_date = st.sidebar.date_input("出生日期", date(1990, 1, 1))
    birth_hour = st.sidebar.slider("出生时辰 (0-23)", 0, 23, 12)
    pillars = get_stem_branch(birth_date.year, birth_date.month, birth_date.day, birth_hour)
else:
    st.sidebar.caption("选择四柱天干地支")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        y_stem = st.selectbox("年干", STEM_LIST, index=8)
        m_stem = st.selectbox("月干", STEM_LIST, index=8)
        d_stem = st.selectbox("日干", STEM_LIST, index=4)
        h_stem = st.selectbox("时干", STEM_LIST, index=9)
    with col2:
        y_branch = st.selectbox("年支", BRANCH_LIST, index=9)
        m_branch = st.selectbox("月支", BRANCH_LIST, index=0)
        d_branch = st.selectbox("日支", BRANCH_LIST, index=2)
        h_branch = st.selectbox("时支", BRANCH_LIST, index=11)
    pillars = [(y_stem, y_branch), (m_stem, m_branch), (d_stem, d_branch), (h_stem, h_branch)]

# Display pillars
pillar_names = ['年柱', '月柱', '日柱', '时柱']
st.sidebar.markdown("---")
st.sidebar.markdown("**四柱:**")
for name, (stem, branch) in zip(pillar_names, pillars):
    st.sidebar.markdown(f"- {name}: {stem}{branch}")

# Sidebar: LLM Settings
st.sidebar.header("🤖 AI 解读 (可选)")
with st.sidebar.expander("🔐 使用 OpenAI API"):
    st.caption("""
    **安全提示**：
    - 建议使用专用 key（非主 key）
    - 设置 spending limit
    - 用完后去 OpenAI 后台 regenerate
    """)
    openai_api_key = st.text_input("API Key", type="password")
    use_llm = st.checkbox("启用 AI 解读", value=bool(openai_api_key), disabled=not openai_api_key)

# ============ Calculate (Shared) ============
w = np.array([1.0, 2.0, 1.0, 1.0])  # Default weights: emphasize month
X = bazi_to_matrix(pillars)
s = X @ w

# Day master element
day_stem = pillars[2][0]
day_element = STEMS[day_stem]
day_element_name = ELEMENTS[day_element]

# Pre-calculate values
sorted_elements = sorted(enumerate(s), key=lambda x: -x[1])
strongest_idx = sorted_elements[0][0]
weakest_idx = sorted_elements[-1][0]
strongest = ELEMENTS_SHORT[strongest_idx]
weakest = ELEMENTS_SHORT[weakest_idx]
day_el = ELEMENTS_SHORT[day_element]

avg_strength = np.mean(s)
day_strength = s[day_element]

# Determine day master status
if day_strength < avg_strength * 0.7:
    day_status = "偏弱"
elif day_strength > avg_strength * 1.5:
    day_status = "偏旺"
else:
    day_status = "中和"

# Smart suggestions based on balance theory
gen_source = (day_element - 1) % 5
gen_target = (day_element + 1) % 5
ctl_source = (day_element + 2) % 5

gen_source_name = ELEMENTS_SHORT[gen_source]
gen_target_name = ELEMENTS_SHORT[gen_target]
ctl_source_name = ELEMENTS_SHORT[ctl_source]

if day_status == "偏弱":
    recommended_intervene = gen_source
elif day_status == "偏旺":
    recommended_intervene = gen_target
else:
    recommended_intervene = gen_source

# ============ Main Content: Tabs ============
tab_experience, tab_theory = st.tabs(["🎯 体验", "🧪 Lab"])

# ============ TAB 1: 体验 (Experience) ============
with tab_experience:
    # Visualization
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("五行雷达图")
        st.plotly_chart(plot_radar(s), use_container_width=True, key="radar_exp")

    with col2:
        st.subheader("五行关系图")
        st.plotly_chart(plot_graph(), use_container_width=True, key="graph_exp")

    # Narrative explanation
    st.subheader("📖 解读")

    st.markdown(f"""
**日主**: {day_stem} ({day_element_name}) — **{day_status}**

五行分布: **{strongest}** 最旺 ({sorted_elements[0][1]:.1f})，**{weakest}** 最弱 ({sorted_elements[-1][1]:.1f})

日主 **{day_el}** 强度 {day_strength:.1f}，平均 {avg_strength:.1f}
""")

    st.markdown("---")
    st.markdown("**调节建议：**")

    if day_status == "偏弱":
        st.markdown(f"""
    日主 {day_el} 偏弱，可考虑：
    - 🔥 **增加 {gen_source_name}**（{gen_source_name} 生 {day_el}，增强日主）
    - 🛡️ **减少 {ctl_source_name}**（{ctl_source_name} 克 {day_el}，削弱日主）
    """)
    elif day_status == "偏旺":
        st.markdown(f"""
    日主 {day_el} 偏旺，可考虑：
    - 💧 **增加 {gen_target_name}**（{day_el} 生 {gen_target_name}，泄日主之气）
    - ⚔️ **增加 {ctl_source_name}**（{ctl_source_name} 克 {day_el}，抑制日主）
    """)
    else:
        st.markdown(f"""
    日主 {day_el} 中和，整体较平衡。可根据具体需求微调。
    """)

    # LLM-powered narrative
    if use_llm and openai_api_key:
        st.markdown("---")
        st.markdown("### 🤖 AI 深度解读")

        col_model, col_btn = st.columns([2, 1])
        with col_model:
            llm_model = st.selectbox(
                "选择模型",
                ["gpt-5.2", "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "o3-mini", "o1"],
                index=0,
                help="gpt-5.2: 最新旗舰 | gpt-4.1: 最强非推理 | o3-mini: 快速推理",
                key="model_selector_exp"
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            generate_btn = st.button("🚀 生成解读", key="llm_btn_exp", use_container_width=True)

        if generate_btn:
            prompt_template = load_prompt_template()
            final_prompt = build_prompt(prompt_template, pillars, s, day_status)
            with st.spinner(f"正在用 {llm_model} 分析..."):
                llm_narrative = generate_llm_narrative(
                    openai_api_key, final_prompt, model=llm_model
                )
                st.markdown(llm_narrative)

# ============ TAB 2: 原理 (Theory) ============
with tab_theory:
    st.caption("如何把 52 万种时间周期位置压缩成 5 个数")
    st.markdown("*换表示（换基/重编码） · 引入语义 · 图结构先验 · 压缩*")

    # Conceptual intro
    with st.expander("💡 这个项目在探索什么？", expanded=False):
        st.markdown("""
        **核心问题：** 如何把 52 万种时间周期位置压缩成 5 个数？

        ```
        时间周期 (52万) ←同构→ 八字符号 (52万) → 五行向量 (5)
                            ↑                   ↑
                      换表示：引入语义+图         压缩
        ```

        - **换表示**：八字编码引入五行语义 + 相生相克图结构
        - **压缩**：22 符号 → 5 五行，权重聚合，图约束

        类似傅立叶变换：同构换基，再截断压缩。

        五行是**载体**，"换表示 + 压缩"是**内核**。

        📖 [完整解释](https://zl190.github.io/blog/zh/wuxing-gnn)
        """)

    # Compression Animation
    st.markdown("### 🔬 换表示 → 压缩")

    bazi_str = f"{pillars[0][0]}{pillars[0][1]} {pillars[1][0]}{pillars[1][1]} {pillars[2][0]}{pillars[2][1]} {pillars[3][0]}{pillars[3][1]}"

    animation_html = f"""
<style>
    .compression-container {{
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        color: white;
        position: relative;
    }}
    #canvas {{
        display: block;
        margin: 0 auto;
    }}
    .overlay {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        pointer-events: none;
    }}
    .stage-label {{
        font-size: 13px;
        color: #888;
        margin-bottom: 5px;
    }}
    .big-text {{
        font-size: 32px;
        font-weight: bold;
        font-family: 'Courier New', monospace;
        color: #fff;
    }}
    .bazi-display {{
        font-size: 32px;
        letter-spacing: 6px;
        color: #4ecdc4;
    }}
    .dim-tag {{
        display: inline-block;
        background: rgba(45, 52, 54, 0.8);
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 11px;
        color: #74b9ff;
        margin-top: 6px;
    }}
    .explanation {{
        font-size: 12px;
        color: #aaa;
        margin-top: 4px;
    }}
    .highlight {{
        color: #f39c12;
    }}
    .btn-replay {{
        margin-top: 10px;
        padding: 6px 16px;
        background: #4ecdc4;
        border: none;
        border-radius: 6px;
        color: #1a1a2e;
        cursor: pointer;
        font-size: 13px;
        pointer-events: auto;
    }}
</style>

<div class="compression-container">
    <canvas id="canvas" width="600" height="300"></canvas>
    <div class="overlay" id="overlay">
        <div id="text-content"></div>
    </div>
</div>

<script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const overlay = document.getElementById('overlay');
    const textContent = document.getElementById('text-content');

    const W = canvas.width, H = canvas.height;
    const centerX = W / 2, centerY = H / 2;

    const numPoints = 800;
    const points = [];
    for (let i = 0; i < numPoints; i++) {{
        points.push({{
            x: Math.random() * W,
            y: Math.random() * H,
            size: Math.random() * 1.5 + 0.5,
            alpha: Math.random() * 0.5 + 0.2
        }});
    }}

    const chosen = {{ x: centerX, y: centerY, size: 3, alpha: 1 }};

    let stage = 0;
    let animFrame = 0;
    let zoomFactor = 1;
    let highlightAlpha = 0;

    function drawPoints(zoom, highlightChosen) {{
        ctx.clearRect(0, 0, W, H);

        points.forEach(p => {{
            const dx = (p.x - centerX) * zoom + centerX;
            const dy = (p.y - centerY) * zoom + centerY;
            if (dx < -50 || dx > W + 50 || dy < -50 || dy > H + 50) return;

            ctx.beginPath();
            ctx.arc(dx, dy, p.size * zoom, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(100, 150, 200, ${{p.alpha / zoom}})`;
            ctx.fill();
        }});

        if (highlightChosen > 0) {{
            const glowSize = 20 + Math.sin(animFrame * 0.1) * 5;
            const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, glowSize);
            gradient.addColorStop(0, `rgba(78, 205, 196, ${{highlightChosen * 0.8}})`);
            gradient.addColorStop(1, 'rgba(78, 205, 196, 0)');
            ctx.beginPath();
            ctx.arc(centerX, centerY, glowSize, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(centerX, centerY, 4, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(78, 205, 196, ${{highlightChosen}})`;
            ctx.fill();
        }}
    }}

    function showText(html) {{
        textContent.innerHTML = html;
    }}

    function runStage0() {{
        stage = 0;
        zoomFactor = 1;
        highlightAlpha = 0;
        showText(`
            <div class="stage-label">时间周期位置</div>
            <div class="big-text">518,400</div>
            <div class="dim-tag">60年 × 12月 × 60日 × 12时</div>
        `);
        drawPoints(1, 0);
        setTimeout(runStage1, 1800);
    }}

    function runStage1() {{
        stage = 1;
        showText(`
            <div class="stage-label">换表示：引入语义 + 图结构</div>
            <div class="big-text">≈ 52 万</div>
            <div class="dim-tag">同构，但每个符号有五行属性</div>
        `);
        drawPoints(1, 0);
        setTimeout(runStage2, 1800);
    }}

    function runStage2() {{
        stage = 2;
        showText(`
            <div class="stage-label">你的八字</div>
            <div class="bazi-display">{bazi_str}</div>
            <div class="dim-tag">52 万中的一个</div>
        `);

        let frame = 0;
        const duration = 60;

        const interval = setInterval(() => {{
            frame++;
            animFrame++;
            const progress = frame / duration;
            const eased = 1 - Math.pow(1 - progress, 2);

            zoomFactor = 1 + eased * 2;
            highlightAlpha = eased;

            drawPoints(zoomFactor, highlightAlpha);

            if (frame >= duration) {{
                clearInterval(interval);
                pulseChosen();
                setTimeout(runStage3, 2000);
            }}
        }}, 25);
    }}

    let pulseInterval = null;
    function pulseChosen() {{
        pulseInterval = setInterval(() => {{
            animFrame++;
            drawPoints(zoomFactor, highlightAlpha);
        }}, 50);
    }}

    function runStage3() {{
        stage = 3;
        if (pulseInterval) clearInterval(pulseInterval);

        let frame = 0;
        const duration = 40;

        const interval = setInterval(() => {{
            frame++;
            const progress = frame / duration;
            zoomFactor = 3 - progress * 2;
            highlightAlpha = 1 - progress * 0.5;
            drawPoints(zoomFactor, highlightAlpha);

            if (frame >= duration) {{
                clearInterval(interval);
            }}
        }}, 25);

        showText(`
            <div class="stage-label">降维：压缩到语义空间</div>
            <div class="big-text" style="font-size:24px;">s = X · w</div>
            <div class="dim-tag">5×4 · 4×1 = 5×1</div>
            <div class="explanation" style="margin-top:8px;"><span class="highlight">52万 → 5</span> 压缩</div>
            <button class="btn-replay" onclick="replay()">↻ 重播</button>
        `);
    }}

    function replay() {{
        if (pulseInterval) clearInterval(pulseInterval);
        runStage0();
    }}

    setTimeout(runStage0, 500);
</script>
"""

    components.html(animation_html, height=380)

    # Step 1: Matrix + Graph
    st.markdown("### 压缩第一步：你的数据 + 共享结构")

    data_col, graph_col = st.columns([1, 1])
    with data_col:
        st.markdown("**你的分布 (5×4 矩阵)**")
        st.caption("个人数据：你的八字映射到五行")
        st.dataframe(pd.DataFrame(X, index=ELEMENTS_SHORT, columns=['年', '月', '日', '时']), height=180)

    with graph_col:
        st.markdown("**共享图谱 (所有人一样)**")
        st.caption("结构先验：2500 年的共识")
        st.markdown("""
    ```
    相生: 木 → 火 → 土 → 金 → 水 → 木
    相克: 木 → 土 → 水 → 火 → 金 → 木
    ```
    """)

    # Step 2: Compress
    st.markdown("### 压缩第二步：5×4 → 5 维向量")
    s_latex = r" \\ ".join([f"{float(v):.1f}" for v in s])
    st.latex(r"s = X \cdot w = \begin{bmatrix}" + s_latex + r"\end{bmatrix}")
    st.caption(f"权重 w = [1, 2, 1, 1]（年/月/日/时，强调月柱）")

    vec_cols = st.columns(5)
    for i, (el, val) in enumerate(zip(ELEMENTS_SHORT, s)):
        with vec_cols[i]:
            st.metric(el, f"{val:.1f}")

    # Quick visualization
    st.markdown("---")
    viz_col1, viz_col2 = st.columns([1, 1])
    with viz_col1:
        st.subheader("五行雷达图")
        st.plotly_chart(plot_radar(s), use_container_width=True, key="radar_theory")
    with viz_col2:
        st.subheader("五行关系图")
        st.plotly_chart(plot_graph(), use_container_width=True, key="graph_theory")

    st.markdown(f"**日主 {day_stem}({day_element_name}) {day_status}** · {strongest} 最旺 · {weakest} 最弱")

    # Deep dive tabs
    st.markdown("---")
    st.subheader("🔬 更多探索")

    sub_tab_exp, sub_tab_formula = st.tabs(["干预实验", "数学公式"])

    with sub_tab_exp:
        st.markdown("#### do(Element += δ): 改变输入，观察传播")
        st.caption("因果推断视角：如果增加某个元素，整个系统会如何响应？")

        # Sidebar params for theory tab
        st.sidebar.header("⚙️ 模型参数")
        lam_g = st.sidebar.slider("λ_生 (相生系数)", 0.0, 1.0, 0.5, 0.05)
        lam_c = st.sidebar.slider("λ_克 (相克系数)", 0.0, 1.0, 0.5, 0.05)
        rho = st.sidebar.slider("ρ (阻尼系数)", 0.0, 1.0, 0.4, 0.05)

        col1, col2 = st.columns([1, 2])

        with col1:
            intervene_element = st.selectbox("干预哪个元素?", ELEMENTS, index=recommended_intervene)
            intervene_idx = ELEMENTS.index(intervene_element)

            observe_element = st.selectbox("观察哪个元素?", ELEMENTS, index=day_element)
            observe_idx = ELEMENTS.index(observe_element)

            delta_values = st.multiselect(
                "δ 值 (干预量)",
                [0, 1, 2, 3, 4, 5],
                default=[0, 1, 2, 4]
            )

            T = st.slider("模拟步数", 10, 50, 25)

        with col2:
            if delta_values:
                trajectories = {}
                for d in sorted(delta_values):
                    h0 = s.copy()
                    h0[intervene_idx] += d
                    traj = simulate(h0, T=T, lam_g=lam_g, lam_c=lam_c, rho=rho)
                    trajectories[d] = traj

                fig = plot_trajectory(trajectories, observe_idx, observe_element, intervene_element)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("请选择至少一个 δ 值")

        if delta_values and len(delta_values) > 1:
            st.markdown(f"""
            **解读**: 增加 **{intervene_element.split()[0]}** (δ) 对 **{observe_element.split()[0]}** 的影响。

            - 曲线上升 → 增强作用（相生传递）
            - 曲线下降 → 抑制作用（相克传递）
            - 稳定性取决于 λ_生, λ_克, ρ 参数
            """)

    with sub_tab_formula:
        st.markdown("#### 形式化：从矩阵到消息传递")

        st.markdown("##### 1. 5×4 分布矩阵 X")
        st.dataframe(pd.DataFrame(X, index=ELEMENTS_SHORT, columns=['年', '月', '日', '时']), height=180)

        st.markdown("##### 2. 权重聚合 s = Xw")
        st.latex(r"w = [1, 2, 1, 1], \quad s = Xw = " + str([round(float(x), 1) for x in s]))

        st.markdown("##### 3. 消息传递更新")
        st.latex(r"h^{(t+1)} = (1-\rho) h^{(t)} + \rho \left( h^{(t)} + \lambda_g A_{gen}^\top h^{(t)} - \lambda_c A_{ctl}^\top h^{(t)} \right)")
        st.latex(r"h^{(t+1)} \leftarrow \max(h^{(t+1)}, 0)")
        st.caption("阻尼 + ReLU 保证收敛和非负")

        st.markdown("##### 4. 邻接矩阵")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**A_gen (相生)**")
            st.dataframe(pd.DataFrame(A_GEN, index=ELEMENTS_SHORT, columns=ELEMENTS_SHORT), height=180)
        with col2:
            st.markdown("**A_ctl (相克)**")
            st.dataframe(pd.DataFrame(A_CTL, index=ELEMENTS_SHORT, columns=ELEMENTS_SHORT), height=180)

    # Aha moment
    if use_llm and openai_api_key:
        st.markdown("---")
        st.markdown("### 🤖 AI 深度解读")

        col_model, col_btn = st.columns([3, 1])
        with col_model:
            llm_model_theory = st.selectbox(
                "模型",
                ["gpt-5.2", "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "o3-mini", "o1"],
                index=0,
                key="model_selector_theory",
                label_visibility="collapsed"
            )
        with col_btn:
            generate_btn_theory = st.button("🚀 生成解读", key="llm_btn_theory", use_container_width=True)

        if generate_btn_theory:
            prompt_template = load_prompt_template()
            final_prompt = build_prompt(prompt_template, pillars, s, day_status)
            with st.spinner(f"{llm_model_theory} 分析中..."):
                llm_narrative = generate_llm_narrative(
                    openai_api_key, final_prompt, model=llm_model_theory
                )
            st.markdown(llm_narrative)

            st.markdown("---")
            st.markdown("""
            > **殊途同归：这就是计算人文**
            >
            > 你刚刚经历了一个完整的形式化建模案例：
            >
            > 1. **表示压缩** — 把 52 万种时间位置压成 5 个数
            > 2. **结构先验** — 引入 2500 年共识的图结构
            > 3. **消息传递** — 用图的视角模拟能量流动
            > 4. **双语解释** — 公式语言 + 人话叙事
            >
            > 五行只是载体。内核是：
            > **用现代框架解构传统系统，看看里面有没有可复用的结构。**
            >
            > 这不是玄学，这是 *计算人文的日常*。
            """)

# ============ Footer ============
st.markdown("---")
st.caption("[🧠 脑洞](https://zl190.github.io/blog/zh/wuxing-gnn) · 用现代框架解构传统系统")
