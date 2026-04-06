# main.py — AI Research Trends Analysis
# Dataset: ArXiv Research Papers (Kaggle/Cornell University)
# Goal: Analyze 20+ years of AI research trends and save processed data for dashboard

import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA (streaming — file is too large to load all at once)
# ─────────────────────────────────────────────────────────────────────────────
print("[1/6] Loading ArXiv data (AI/ML papers only)...")

AI_CATEGORIES = [
    'cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE',
    'cs.RO', 'stat.ML'
]

CATEGORY_NAMES = {
    'cs.AI': 'Artificial Intelligence',
    'cs.LG': 'Machine Learning',
    'cs.CL': 'Natural Language Processing',
    'cs.CV': 'Computer Vision',
    'cs.NE': 'Neural & Evolutionary Computing',
    'cs.RO': 'Robotics',
    'stat.ML': 'Statistical ML'
}

def extract_year(date_str):
    """Extract year from various date formats."""
    if not date_str:
        return None
    match = re.search(r'(\d{4})', str(date_str))
    return int(match.group(1)) if match else None

def get_primary_category(categories_str):
    """Get the first listed category."""
    if not categories_str:
        return None
    return str(categories_str).split()[0]

def is_ai_paper(categories_str):
    """Check if paper belongs to any AI category."""
    if not categories_str:
        return False
    return any(cat in str(categories_str) for cat in AI_CATEGORIES)

# Stream through the large JSON file
records = []
with open('data/raw/arxiv-metadata-oai-snapshot.json', 'r') as f:
    for i, line in enumerate(f):
        try:
            paper = json.loads(line)
            if is_ai_paper(paper.get('categories', '')):
                year = extract_year(paper.get('update_date', ''))
                if year and 2000 <= year <= 2023:
                    records.append({
                        'id':          paper.get('id', ''),
                        'title':       paper.get('title', '').replace('\n', ' ').strip(),
                        'abstract':    paper.get('abstract', '').replace('\n', ' ').strip(),
                        'categories':  paper.get('categories', ''),
                        'primary_cat': get_primary_category(paper.get('categories', '')),
                        'year':        year,
                        'authors':     paper.get('authors', ''),
                    })
        except:
            continue
        if i % 500000 == 0:
            print(f"  Processed {i:,} lines, found {len(records):,} AI papers so far...")

df = pd.DataFrame(records)
print(f"\n  Total AI papers loaded: {len(df):,}")
print(f"  Year range: {df['year'].min()} - {df['year'].max()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CLEAN & ENRICH
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Cleaning and enriching...")

# Map category codes to readable names
df['category_name'] = df['primary_cat'].map(CATEGORY_NAMES).fillna('Other AI')

# Count authors
df['author_count'] = df['authors'].apply(
    lambda x: len(str(x).split(',')) if x else 1)
df['author_count'] = df['author_count'].clip(1, 50)

# Collaboration flag
df['is_collaborative'] = (df['author_count'] > 1).astype(int)

# Clean title for NLP
df['title_clean'] = df['title'].apply(
    lambda x: re.sub(r'[^a-zA-Z\s]', ' ', str(x)).lower().strip())

print(f"  Clean shape: {df.shape}")
print(f"  Category breakdown:\n{df['category_name'].value_counts()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. AGGREGATE — ANNUAL TRENDS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Computing aggregations...")

# Papers per year
annual = (df.groupby('year')
          .agg(total_papers=('id', 'count'),
               avg_authors=('author_count', 'mean'),
               collaborative_pct=('is_collaborative', 'mean'))
          .reset_index())
annual['collaborative_pct'] = (annual['collaborative_pct'] * 100).round(1)
annual['avg_authors'] = annual['avg_authors'].round(2)

# Papers per year per category
cat_annual = (df.groupby(['year', 'category_name'])
              .size().reset_index(name='paper_count'))

# Top keywords per decade
def get_decade_keywords(df, start_year, end_year, top_n=50):
    from sklearn.feature_extraction.text import TfidfVectorizer
    subset = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    texts = subset['title_clean'].dropna().tolist()
    if not texts:
        return {}
    stop_words = ['using', 'based', 'via', 'new', 'deep', 'learning',
                  'neural', 'network', 'model', 'method', 'approach']
    tfidf = TfidfVectorizer(max_features=200, stop_words='english',
                             ngram_range=(1, 2))
    tfidf.fit(texts)
    scores = dict(zip(tfidf.get_feature_names_out(),
                      tfidf.idf_))
    sorted_terms = sorted(scores.items(), key=lambda x: -x[1])
    return dict(sorted_terms[:top_n])

decades = {
    '2000s': get_decade_keywords(df, 2000, 2009),
    '2010s': get_decade_keywords(df, 2010, 2019),
    '2020s': get_decade_keywords(df, 2020, 2023),
}

# Save all processed data
annual.to_csv('data/processed/annual_trends.csv', index=False)
cat_annual.to_csv('data/processed/category_annual.csv', index=False)
df.to_csv('data/processed/papers_clean.csv', index=False)
print("  Saved processed data to data/processed/")

# ─────────────────────────────────────────────────────────────────────────────
# 4. STATIC VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Generating static charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: Total papers per year
axes[0,0].fill_between(annual['year'], annual['total_papers'],
                        alpha=0.4, color='steelblue')
axes[0,0].plot(annual['year'], annual['total_papers'],
               color='steelblue', linewidth=2)
axes[0,0].set_title('AI Research Papers Published per Year')
axes[0,0].set_xlabel('Year')
axes[0,0].set_ylabel('Number of Papers')

# Annotate key moments
annotations = {2012: 'AlexNet', 2017: 'Transformers', 2020: 'GPT-3'}
for year, label in annotations.items():
    if year in annual['year'].values:
        y_val = annual[annual['year'] == year]['total_papers'].values[0]
        axes[0,0].annotate(label, xy=(year, y_val),
                           xytext=(year-1, y_val + 2000),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontsize=8, color='red')

# Chart 2: Papers by category (top 5)
top_cats = df['category_name'].value_counts().head(5).index
cat_pivot = cat_annual[cat_annual['category_name'].isin(top_cats)].pivot(
    index='year', columns='category_name', values='paper_count').fillna(0)
cat_pivot.plot(ax=axes[0,1], linewidth=2)
axes[0,1].set_title('Papers by AI Category Over Time')
axes[0,1].set_xlabel('Year')
axes[0,1].set_ylabel('Papers')
axes[0,1].legend(fontsize=7)

# Chart 3: Average authors per paper over time
axes[1,0].plot(annual['year'], annual['avg_authors'],
               color='coral', linewidth=2, marker='o', markersize=4)
axes[1,0].set_title('Average Authors per Paper Over Time')
axes[1,0].set_xlabel('Year')
axes[1,0].set_ylabel('Avg Authors')
axes[1,0].axhline(y=annual['avg_authors'].mean(), color='gray',
                   linestyle='--', alpha=0.7)

# Chart 4: Category distribution pie
cat_counts = df['category_name'].value_counts()
axes[1,1].pie(cat_counts.values, labels=cat_counts.index,
              autopct='%1.1f%%', startangle=140)
axes[1,1].set_title('AI Paper Distribution by Category')

plt.suptitle('ArXiv AI Research Trends (2000-2023)', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/research_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/research_trends.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. WORD CLOUDS BY DECADE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Generating word clouds by decade...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (decade, words) in zip(axes, decades.items()):
    if words:
        wc = WordCloud(width=600, height=400, background_color='white',
                       colormap='Blues', max_words=60).generate_from_frequencies(words)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Top AI Terms — {decade}', fontsize=13, fontweight='bold')

plt.suptitle('How AI Research Vocabulary Evolved by Decade', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/decade_wordclouds.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/decade_wordclouds.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY STATS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Summary statistics...")
print(f"  Total AI papers (2000-2023): {len(df):,}")
print(f"  Fastest growing year: {annual.loc[annual['total_papers'].diff().idxmax(), 'year']}")
print(f"  Most common category: {df['category_name'].value_counts().index[0]}")
print(f"  Avg authors per paper: {df['author_count'].mean():.2f}")
print(f"  Collaborative papers: {df['is_collaborative'].mean()*100:.1f}%")

print("\n✅ Pipeline complete. Run dashboard.py next for the interactive dashboard!")