import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
from itertools import islice
from wordcloud import WordCloud
import os

# Create output directories if they don't exist
os.makedirs('lexical_eda_out/tables', exist_ok=True)
os.makedirs('lexical_eda_out/figures', exist_ok=True)

# ========== DATA LOADING & SETUP ==========
print("Loading dataset...")
df = pd.read_csv('datasets/dontpatronizeme_pcl.tsv', sep='\t', skiprows=4, header=None, 
                  names=["id","par_id","keyword","country","text","label"])
df['y'] = (df['label'] >= 2).astype(int)

print(f"Dataset loaded: {len(df)} rows")
print(f"Class distribution: {df['y'].value_counts().to_dict()}")

# ========== TOKENIZATION & PREPROCESSING ==========
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tokenize(text):
    """Tokenize text to lowercase words"""
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())

# English stop words (common filler words)
ENGLISH_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any',
    'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'but', 'by', 'can', 'can\'t', 'can\'ve', 'could', 'couldn\'t',
    'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have',
    'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s',
    'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll',
    'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself',
    'just', 'k', 'me', 'might', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor',
    'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 's', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll',
    'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 't', 'than', 'that', 'that\'s',
    'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these',
    'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'to', 'too',
    'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re',
    'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s',
    'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would',
    'wouldn\'t', 'y', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours',
    'yourself', 'yourselves'
}

# Tokenize all text
print("Tokenizing texts...")
df['tokens'] = df['text'].apply(tokenize)

# ========== N-GRAM EXTRACTION ==========
def extract_ngrams(tokens, n):
    """Extract n-grams from a list of tokens"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def extract_ngrams_no_stopwords(tokens, n):
    """Extract n-grams excluding stop words"""
    filtered = [t for t in tokens if t not in ENGLISH_STOPWORDS]
    return [tuple(filtered[i:i+n]) for i in range(len(filtered) - n + 1)]

print("\nExtracting n-grams...")
# Extract bigrams and trigrams
df['bigrams'] = df['tokens'].apply(lambda x: extract_ngrams(x, 2) if len(x) >= 2 else [])
df['trigrams'] = df['tokens'].apply(lambda x: extract_ngrams(x, 3) if len(x) >= 3 else [])
df['bigrams_no_sw'] = df['tokens'].apply(lambda x: extract_ngrams_no_stopwords(x, 2) if len(x) >= 2 else [])
df['trigrams_no_sw'] = df['tokens'].apply(lambda x: extract_ngrams_no_stopwords(x, 3) if len(x) >= 3 else [])

# ========== STOP WORD DENSITY ANALYSIS ==========
print("\nCalculating stop word statistics...")

def calculate_stopword_density(tokens):
    """Calculate percentage of stop words in tokens"""
    if len(tokens) == 0:
        return 0
    stop_count = sum(1 for t in tokens if t in ENGLISH_STOPWORDS)
    return (stop_count / len(tokens)) * 100

df['stopword_density'] = df['tokens'].apply(calculate_stopword_density)

# Overall statistics
overall_density = df['stopword_density'].mean()
class0_density = df[df['y'] == 0]['stopword_density'].mean()
class1_density = df[df['y'] == 1]['stopword_density'].mean()

print(f"Overall stop word density: {overall_density:.2f}%")
print(f"Class 0 (No PCL) stop word density: {class0_density:.2f}%")
print(f"Class 1 (PCL) stop word density: {class1_density:.2f}%")

# ========== N-GRAM FREQUENCY ANALYSIS ==========
print("\nAnalyzing n-gram frequencies...")

# Bigrams
all_bigrams = [bigram for bigrams in df['bigrams'] for bigram in bigrams]
class0_bigrams = [bigram for bigrams in df[df['y'] == 0]['bigrams'] for bigram in bigrams]
class1_bigrams = [bigram for bigrams in df[df['y'] == 1]['bigrams'] for bigram in bigrams]

bigram_freq_all = Counter(all_bigrams)
bigram_freq_c0 = Counter(class0_bigrams)
bigram_freq_c1 = Counter(class1_bigrams)

# Trigrams
all_trigrams = [trigram for trigrams in df['trigrams'] for trigram in trigrams]
class0_trigrams = [trigram for trigrams in df[df['y'] == 0]['trigrams'] for trigram in trigrams]
class1_trigrams = [trigram for trigrams in df[df['y'] == 1]['trigrams'] for trigram in trigrams]

trigram_freq_all = Counter(all_trigrams)
trigram_freq_c0 = Counter(class0_trigrams)
trigram_freq_c1 = Counter(class1_trigrams)

print(f"Total bigrams: {len(all_bigrams)}, unique: {len(bigram_freq_all)}")
print(f"Total trigrams: {len(all_trigrams)}, unique: {len(trigram_freq_all)}")

# ========== HELPER FUNCTIONS FOR OUTPUT ==========
def ngrams_to_dataframe(ngram_counter, top_n=50):
    """Convert Counter to DataFrame with frequencies and percentages"""
    total = sum(ngram_counter.values())
    data = []
    for ngram, count in ngram_counter.most_common(top_n):
        text = ' '.join(ngram)
        percentage = (count / total) * 100
        data.append({'ngram': text, 'frequency': count, 'percentage': f'{percentage:.4f}'})
    return pd.DataFrame(data)

def save_ngram_tables(bigram_freq, trigram_freq, suffix, output_dir='lexical_eda_out/tables'):
    """Save n-gram frequencies to CSV and LaTeX files"""
    
    # Bigrams
    bigram_df = ngrams_to_dataframe(bigram_freq, top_n=100)
    bigram_csv = f'{output_dir}/top_2gram_{suffix}.csv'
    bigram_tex = f'{output_dir}/top_2gram_{suffix}.tex'
    
    bigram_df.to_csv(bigram_csv, index=False)
    bigram_df.to_latex(bigram_tex, index=False)
    print(f"Saved {bigram_csv}")
    
    # Trigrams
    trigram_df = ngrams_to_dataframe(trigram_freq, top_n=100)
    trigram_csv = f'{output_dir}/top_3gram_{suffix}.csv'
    trigram_tex = f'{output_dir}/top_3gram_{suffix}.tex'
    
    trigram_df.to_csv(trigram_csv, index=False)
    trigram_df.to_latex(trigram_tex, index=False)
    print(f"Saved {trigram_csv}")

# ========== GENERATE N-GRAM TABLES ==========
print("\nGenerating n-gram tables...")
save_ngram_tables(bigram_freq_all, trigram_freq_all, 'overall')
save_ngram_tables(bigram_freq_c0, trigram_freq_c0, 'class0')
save_ngram_tables(bigram_freq_c1, trigram_freq_c1, 'class1')

# ========== WORD CLOUDS ==========
print("\nGenerating word clouds...")

# Overall corpus word cloud
all_tokens = [token for tokens in df['tokens'] for token in tokens]
all_tokens_text = ' '.join(all_tokens)

wc = WordCloud(width=800, height=600, background_color='white').generate(all_tokens_text)
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Overall Corpus')
plt.tight_layout(pad=0)
plt.savefig('lexical_eda_out/figures/wordcloud_overall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved wordcloud_overall.png")

# Class-separated word clouds
for class_id, class_name in [(0, 'class0'), (1, 'class1')]:
    class_tokens = [token for tokens in df[df['y'] == class_id]['tokens'] for token in tokens]
    class_text = ' '.join(class_tokens)
    
    wc = WordCloud(width=800, height=600, background_color='white').generate(class_text)
    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - {class_name.upper()}')
    plt.tight_layout(pad=0)
    plt.savefig(f'lexical_eda_out/figures/wordcloud_{class_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved wordcloud_{class_name}.png")

# Content words (no stop words) word cloud
content_tokens = [token for tokens in df['tokens'] for token in tokens if token not in ENGLISH_STOPWORDS]
content_text = ' '.join(content_tokens)

wc = WordCloud(width=800, height=600, background_color='white').generate(content_text)
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Content Words (Stop Words Excluded)')
plt.tight_layout(pad=0)
plt.savefig('lexical_eda_out/figures/wordcloud_content_only.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved wordcloud_content_only.png")

# ========== FREQUENCY DISTRIBUTION PLOTS ==========
print("\nGenerating frequency distribution plots...")

# Bigram frequency distribution (top 20)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Overall bigrams
top_bigrams = bigram_freq_all.most_common(20)
bigram_names = [' '.join(bg) for bg, _ in top_bigrams]
bigram_freqs = [freq for _, freq in top_bigrams]

axes[0].barh(bigram_names, bigram_freqs, color='steelblue')
axes[0].set_xlabel('Frequency')
axes[0].set_title('Top 20 Bigrams - Overall')
axes[0].invert_yaxis()

# Class 0 bigrams
top_bigrams_c0 = bigram_freq_c0.most_common(20)
bigram_names_c0 = [' '.join(bg) for bg, _ in top_bigrams_c0]
bigram_freqs_c0 = [freq for _, freq in top_bigrams_c0]

axes[1].barh(bigram_names_c0, bigram_freqs_c0, color='coral')
axes[1].set_xlabel('Frequency')
axes[1].set_title('Top 20 Bigrams - Class 0 (No PCL)')
axes[1].invert_yaxis()

# Class 1 bigrams
top_bigrams_c1 = bigram_freq_c1.most_common(20)
bigram_names_c1 = [' '.join(bg) for bg, _ in top_bigrams_c1]
bigram_freqs_c1 = [freq for _, freq in top_bigrams_c1]

axes[2].barh(bigram_names_c1, bigram_freqs_c1, color='seagreen')
axes[2].set_xlabel('Frequency')
axes[2].set_title('Top 20 Bigrams - Class 1 (PCL)')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('lexical_eda_out/figures/bigram_frequencies.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved bigram_frequencies.png")

# Trigram frequency distribution (top 20)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Overall trigrams
top_trigrams = trigram_freq_all.most_common(20)
trigram_names = [' '.join(tg) for tg, _ in top_trigrams]
trigram_freqs = [freq for _, freq in top_trigrams]

axes[0].barh(trigram_names, trigram_freqs, color='steelblue')
axes[0].set_xlabel('Frequency')
axes[0].set_title('Top 20 Trigrams - Overall')
axes[0].invert_yaxis()

# Class 0 trigrams
top_trigrams_c0 = trigram_freq_c0.most_common(20)
trigram_names_c0 = [' '.join(tg) for tg, _ in top_trigrams_c0]
trigram_freqs_c0 = [freq for _, freq in top_trigrams_c0]

axes[1].barh(trigram_names_c0, trigram_freqs_c0, color='coral')
axes[1].set_xlabel('Frequency')
axes[1].set_title('Top 20 Trigrams - Class 0 (No PCL)')
axes[1].invert_yaxis()

# Class 1 trigrams
top_trigrams_c1 = trigram_freq_c1.most_common(20)
trigram_names_c1 = [' '.join(tg) for tg, _ in top_trigrams_c1]
trigram_freqs_c1 = [freq for _, freq in top_trigrams_c1]

axes[2].barh(trigram_names_c1, trigram_freqs_c1, color='seagreen')
axes[2].set_xlabel('Frequency')
axes[2].set_title('Top 20 Trigrams - Class 1 (PCL)')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('lexical_eda_out/figures/trigram_frequencies.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved trigram_frequencies.png")

# ========== STOP WORD DENSITY COMPARISON ==========
print("\nGenerating stop word density plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Density by class (bar chart)
classes = ['No PCL (0)', 'PCL (1)']
densities = [class0_density, class1_density]
colors = ['coral', 'seagreen']

axes[0].bar(classes, densities, color=colors, alpha=0.7)
axes[0].set_ylabel('Stop Word Density (%)')
axes[0].set_title('Average Stop Word Density by Class')
axes[0].set_ylim([0, max(densities) * 1.2])
for i, v in enumerate(densities):
    axes[0].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')

# Distribution of stop word density
axes[1].hist(df[df['y'] == 0]['stopword_density'], bins=30, alpha=0.6, label='No PCL', color='coral')
axes[1].hist(df[df['y'] == 1]['stopword_density'], bins=30, alpha=0.6, label='PCL', color='seagreen')
axes[1].set_xlabel('Stop Word Density (%)')
axes[1].set_ylabel('Number of Paragraphs')
axes[1].set_title('Distribution of Stop Word Density')
axes[1].legend()

plt.tight_layout()
plt.savefig('lexical_eda_out/figures/stopword_density.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved stopword_density.png")

# ========== SUMMARY STATISTICS ==========
print("\n" + "="*60)
print("LEXICAL ANALYSIS SUMMARY")
print("="*60)

print(f"\n📊 DATASET STATISTICS:")
print(f"  Total paragraphs: {len(df)}")
print(f"  Class 0 (No PCL): {(df['y'] == 0).sum()} ({(df['y'] == 0).sum()/len(df)*100:.1f}%)")
print(f"  Class 1 (PCL): {(df['y'] == 1).sum()} ({(df['y'] == 1).sum()/len(df)*100:.1f}%)")

print(f"\n🔤 TOKEN STATISTICS:")
print(f"  Total tokens: {len(all_tokens)}")
print(f"  Unique tokens (vocabulary size): {len(set(all_tokens))}")
print(f"  Average tokens per paragraph: {len(all_tokens) / len(df):.2f}")

print(f"\n🛑 STOP WORD ANALYSIS:")
print(f"  Overall stop word density: {overall_density:.2f}%")
print(f"  Class 0 stop word density: {class0_density:.2f}%")
print(f"  Class 1 stop word density: {class1_density:.2f}%")
print(f"  Difference (Class 1 - Class 0): {class1_density - class0_density:+.2f}%")

print(f"\n2️⃣ BIGRAM STATISTICS:")
print(f"  Total bigrams: {len(all_bigrams)}")
print(f"  Unique bigrams: {len(bigram_freq_all)}")
print(f"  Top 5 bigrams overall:")
for i, (bigram, freq) in enumerate(bigram_freq_all.most_common(5), 1):
    print(f"    {i}. {' '.join(bigram)}: {freq}")

print(f"\n3️⃣ TRIGRAM STATISTICS:")
print(f"  Total trigrams: {len(all_trigrams)}")
print(f"  Unique trigrams: {len(trigram_freq_all)}")
print(f"  Top 5 trigrams overall:")
for i, (trigram, freq) in enumerate(trigram_freq_all.most_common(5), 1):
    print(f"    {i}. {' '.join(trigram)}: {freq}")

print(f"\n📁 OUTPUT FILES GENERATED:")
print(f"  ✓ N-gram CSV tables (top_*gram_*.csv)")
print(f"  ✓ N-gram LaTeX tables (top_*gram_*.tex)")
print(f"  ✓ Word clouds (wordcloud_*.png)")
print(f"  ✓ Frequency distribution plots (bigram/trigram_frequencies.png)")
print(f"  ✓ Stop word density plots (stopword_density.png)")
print("="*60)