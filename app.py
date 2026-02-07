# -*- coding: utf-8 -*-
"""
Wuxing Demo - Five Elements Visualization
五行可视化与解读

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
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

st.title("计算人文")
st.markdown("### 五行")
st.caption("输入生辰，看五行分布，生成 AI 解读")

# Sidebar: Input
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

# Calculate
w = np.array([1.0, 2.0, 1.0, 1.0])  # Default weights: emphasize month
X = bazi_to_matrix(pillars)
s = X @ w

# Day master element
day_stem = pillars[2][0]
day_element = STEMS[day_stem]
day_element_name = ELEMENTS[day_element]

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("五行雷达图")
    st.plotly_chart(plot_radar(s), use_container_width=True)

with col2:
    st.subheader("五行关系图")
    st.plotly_chart(plot_graph(), use_container_width=True)

# Narrative explanation
st.subheader("📖 解读")

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

st.markdown(f"""
**日主**: {day_stem} ({day_element_name}) — **{day_status}**

五行分布: **{strongest}** 最旺 ({sorted_elements[0][1]:.1f})，**{weakest}** 最弱 ({sorted_elements[-1][1]:.1f})

日主 **{day_el}** 强度 {day_strength:.1f}，平均 {avg_strength:.1f}
""")

# Smart suggestions based on balance theory
gen_source = (day_element - 1) % 5  # element that generates day master
gen_target = (day_element + 1) % 5  # element that day master generates (drains)
ctl_source = (day_element + 2) % 5  # element that controls day master

gen_source_name = ELEMENTS_SHORT[gen_source]
gen_target_name = ELEMENTS_SHORT[gen_target]
ctl_source_name = ELEMENTS_SHORT[ctl_source]

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

# LLM-powered narrative (if enabled)
if use_llm and openai_api_key:
    st.markdown("---")
    st.markdown("### 🤖 AI 深度解读")

    # Model selector and generate button in same row
    col_model, col_btn = st.columns([2, 1])
    with col_model:
        llm_model = st.selectbox(
            "选择模型",
            ["gpt-5.2", "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "o3-mini", "o1"],
            index=0,
            help="gpt-5.2: 最新旗舰 | gpt-4.1: 最强非推理 | o3-mini: 快速推理",
            key="model_selector"
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)  # align with selectbox
        generate_btn = st.button("🚀 生成解读", key="llm_btn", use_container_width=True)

    if generate_btn:
        # Load prompt from file and build final prompt
        prompt_template = load_prompt_template()
        final_prompt = build_prompt(prompt_template, pillars, s, day_status)
        with st.spinner(f"正在用 {llm_model} 分析..."):
            llm_narrative = generate_llm_narrative(
                openai_api_key, final_prompt, model=llm_model
            )
            st.markdown(llm_narrative)

# Footer
st.markdown("---")
st.caption("[📐 计算人文版](/DS) · [🧠 原理](https://zl190.github.io/blog/zh/wuxing-gnn) · 用现代框架解构传统系统")
