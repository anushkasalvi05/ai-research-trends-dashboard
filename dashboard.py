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
    papers   = pd.read_csv('data/processed/papers_clean.csv')
    return annual, cat_ann, papers

annual, cat_annual, papers = load_data()

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
col1.metric("Total Papers", f"{len(papers_filtered):,}")
col2.metric("Categories", f"{papers_filtered['category_name'].nunique()}")
col3.metric("Avg Authors/Paper", f"{papers_filtered['author_count'].mean():.2f}")
col4.metric("Collaborative Papers", f"{papers_filtered['is_collaborative'].mean()*100:.1f}%")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: PAPER GROWTH OVER TIME
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📈 AI Paper Growth Over Time")

annual_filtered2 = papers_filtered.groupby('year').size().reset_index(name='total_papers')

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
    cat_dist = papers_filtered['category_name'].value_counts().reset_index()
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
    fig4 = px.line(collab, x='year', y='avg_authors',
                   title='Average Authors per Paper',
                   template='plotly_white', height=350,
                   markers=True)
    fig4.update_traces(line_color='coral', line_width=3)
    st.plotly_chart(fig4, use_container_width=True)

with col_d:
    fig5 = px.bar(collab, x='year', y='collab_pct',
                  title='% Collaborative Papers per Year',
                  template='plotly_white', height=350,
                  color='collab_pct',
                  color_continuous_scale='Blues')
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# WORD CLOUD BY DECADE
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("☁️ How AI Research Vocabulary Evolved")

decade_col1, decade_col2, decade_col3 = st.columns(3)
decade_map = {
    '2000s': (2007, 2009),
    '2010s': (2010, 2019),
    '2020s': (2020, 2023)
}

for col, (decade, (start, end)) in zip(
    [decade_col1, decade_col2, decade_col3], decade_map.items()
):
    subset = papers[(papers['year'] >= start) & (papers['year'] <= end)]
    text = ' '.join(subset['title_clean'].dropna().tolist())
    if text.strip():
        wc = WordCloud(width=500, height=300, background_color='white',
                       colormap='Blues', max_words=60).generate(text)
        fig_wc, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        col.markdown(f"**{decade}**")
        col.pyplot(fig_wc)
        plt.close()

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
---
**Built by Anushka Rajesh Salvi** | MS Data Science, George Washington University |
[LinkedIn](https://linkedin.com/in/anushka-rajesh-salvi) · [GitHub](https://github.com/anushkasalvi05)
""")