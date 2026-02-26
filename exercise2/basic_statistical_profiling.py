import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter

# Load dataset
df = pd.read_csv('datasets/dontpatronizeme_pcl.tsv', sep='\t', skiprows=4, header=None, names=["id","par_id","keyword","country","text","label"])

df['y'] = (df['label'] >= 2).astype(int)

######## TOKEN COUNT ANALYSIS #######
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tokenize(text):
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())
    
df['tokens'] = df['text'].apply(tokenize)
df['length'] = df['tokens'].apply(len)

token_stats = {
    "mean_length": df["length"].mean(),
    "median_length": df["length"].median(),
    "min_length": df["length"].min(),
    "max_length": df["length"].max(),
    "p95_length": df["length"].quantile(0.95)
}

plt.figure(figsize=(8,5))

plt.hist(df[df["y"]==0]["length"],
         bins=50,
         alpha=0.6,
         label="No PCL")

plt.hist(df[df["y"]==1]["length"],
         bins=50,
         alpha=0.6,
         label="PCL")

plt.xlabel("Token Count")
plt.ylabel("Number of Paragraphs")
plt.title("Token Length Distribution")
plt.legend()

plt.savefig("length_distribution.pdf")
plt.show()


######## Vocabulary Size Analysis #######

all_tokens = [token for tokens in df['tokens'] for token in tokens]
vocab = set(all_tokens)
vocab_size = len(vocab)
total_tokens = len(all_tokens)
print(f"Vocabulary Size: {vocab_size}")
print(f"Total Tokens: {total_tokens}")

# Most frequent words
print("20 Most frequent words: ", Counter(all_tokens).most_common(20))


######## Class Distribution #######

class_counts = df["y"].value_counts().sort_index()

print(class_counts)

class_percent = df["y"].value_counts(normalize=True)*100
print(class_percent)

class_table = pd.DataFrame({
    "Class":["No PCL (0)","PCL (1)"],
    "Count":[class_counts[0], class_counts[1]],
    "Percentage":[class_percent[0], class_percent[1]]
})

print(class_table)

plt.figure(figsize=(6,4))
df.boxplot(column="length", by="y")
plt.title("Length by Class")
plt.suptitle("")
plt.xlabel("Class (0=No PCL, 1=PCL)")
plt.ylabel("Token Length")

plt.savefig("length_boxplot.pdf")
plt.show()


