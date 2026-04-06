# dashboard.py — AI Research Trends Interactive Dashboard
# Run with: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Trends Dashboard",
    page_icon="🤖",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    annual   = pd.read_csv('data/processed/annual_trends.csv')
    cat_ann  = pd.read_csv('data/processed/category_annual.csv')
    return annual, cat_ann

annual, cat_annual = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🤖 AI Research Trends Dashboard")
st.markdown("Analyzing **312,925 AI research papers** from ArXiv (2007–2023)")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("Filters")
year_range = st.sidebar.slider(
    "Year Range",
    min_value=int(annual['year'].min()),
    max_value=int(annual['year'].max()),
    value=(2007, 2023)
)

all_cats = sorted(papers['category_name'].unique().tolist())
selected_cats = st.sidebar.multiselect(
    "Select Categories",
    options=all_cats,
    default=all_cats
)

# Apply filters
annual_filtered = annual[
    (annual['year'] >= year_range[0]) &
    (annual['year'] <= year_range[1])
]
papers_filtered = papers[
    (papers['year'] >= year_range[0]) &
    (papers['year'] <= year_range[1]) &
    (papers['category_name'].isin(selected_cats))
]
cat_filtered = cat_annual[
    (cat_annual['year'] >= year_range[0]) &
    (cat_annual['year'] <= year_range[1]) &
    (cat_annual['category_name'].isin(selected_cats))
]

# ─────────────────────────────────────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
total = cat_filtered['paper_count'].sum()
col1.metric("Total Papers", f"{int(total):,}")
col2.metric("Categories", f"{cat_filtered['category_name'].nunique()}")
col3.metric("Avg Authors/Paper", "3.78")
col4.metric("Collaborative Papers", "78.9%")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: PAPER GROWTH OVER TIME
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📈 AI Paper Growth Over Time")

annual_filtered2 = cat_filtered.groupby('year')['paper_count'].sum().reset_index()
annual_filtered2.columns = ['year', 'total_papers']

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=annual_filtered2['year'],
    y=annual_filtered2['total_papers'],
    mode='lines+markers',
    fill='tozeroy',
    fillcolor='rgba(70, 130, 180, 0.2)',
    line=dict(color='steelblue', width=3),
    marker=dict(size=6),
    hovertemplate='Year: %{x}<br>Papers: %{y:,}<extra></extra>'
))

# Key milestone annotations
milestones = {2012: 'AlexNet', 2017: 'Transformers', 2020: 'GPT-3', 2022: 'ChatGPT'}
for year, label in milestones.items():
    if year_range[0] <= year <= year_range[1]:
        y_val = annual_filtered2[annual_filtered2['year'] == year]['total_papers']
        if not y_val.empty:
            fig1.add_annotation(
                x=year, y=y_val.values[0],
                text=f"📍{label}",
                showarrow=True, arrowhead=2,
                arrowcolor='red', font=dict(size=11, color='red'),
                ay=-40
            )

fig1.update_layout(
    xaxis_title='Year', yaxis_title='Number of Papers',
    template='plotly_white', height=400,
    hovermode='x unified'
)
st.plotly_chart(fig1, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 & 3: SIDE BY SIDE
# ─────────────────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Papers by Category Over Time")
    fig2 = px.area(
        cat_filtered, x='year', y='paper_count',
        color='category_name',
        template='plotly_white',
        labels={'paper_count': 'Papers', 'year': 'Year',
                'category_name': 'Category'},
        height=400
    )
    fig2.update_layout(legend=dict(font=dict(size=10)))
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.subheader("🥧 Category Distribution")
    cat_dist = cat_filtered.groupby('category_name')['paper_count'].sum().reset_index()
    cat_dist.columns = ['category', 'count']
    fig3 = px.pie(
        cat_dist, values='count', names='category',
        template='plotly_white', height=400,
        hole=0.3
    )
    fig3.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CHART 4: COLLABORATION TRENDS
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🤝 Collaboration Trends Over Time")

collab = papers_filtered.groupby('year').agg(
    avg_authors=('author_count', 'mean'),
    collab_pct=('is_collaborative', 'mean')
).reset_index()
collab['collab_pct'] = collab['collab_pct'] * 100

col_c, col_d = st.columns(2)
with col_c:
    fig4 = px.line(annual_filtered, x='year', y='avg_authors',
                   title='Average Authors per Paper',
                   template='plotly_white', height=350,
                   markers=True)
    fig4.update_traces(line_color='coral', line_width=3)
    st.plotly_chart(fig4, use_container_width=True)

with col_d:
    fig5 = px.bar(annual_filtered, x='year', y='collaborative_pct',
                  title='% Collaborative Papers per Year',
                  template='plotly_white', height=350,
                  color='collaborative_pct',
                  color_continuous_scale='Blues')
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
---
**Built by Anushka Rajesh Salvi** | MS Data Science, George Washington University |
[LinkedIn](https://linkedin.com/in/anushka-rajesh-salvi) · [GitHub](https://github.com/anushkasalvi05)
""")