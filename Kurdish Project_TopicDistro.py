#sentiment analysis and topic frequency

from bertopic import BERTopic
import pandas as pd
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
from transformers import pipeline
import os

dataset = glob.glob("general_event\*.csv")
df_list = []
for file in dataset:
    try:
        df = pd.read_csv(file)
        if not {"content", "date"}.issubset(df.columns):
            print(f"Skipping {file} — requires 'content' and 'date' columns.")
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["content", "date"])

        df_list.append(df)

    except Exception as e:
        print(f"Error reading {file}: {e}")

grouped_texts = []

for df in df_list:
    df = df.sort_values(by="date")
    df["group"] = df["date"].dt.floor("5min")
    grouped = df.groupby("group")["content"].apply(lambda msgs: " ".join(msgs)).tolist()
    grouped_texts.extend(grouped)

print("Training BERTopic model...")
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(grouped_texts)

print("Running sentiment analysis...")
sentiment_model = pipeline("sentiment-analysis")

batch_size = 32
sentiments = []

#Intertopic distance map;
print("\nTopic Info:")
print(topic_model.get_topic_info())
topic_model.visualize_topics().show()

#top 10 topics
topic_info = topic_model.get_topic_info()
top_topics = topic_info[topic_info.Topic >= 0].nlargest(10, 'Count')

plt.figure(figsize=(10, 5))
plt.barh(top_topics['Name'], top_topics['Count'])
plt.xlabel("Number of Documents")
plt.title("Top 10 Topics by Frequency")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
plt.savefig("Top Topics By Freq.png")


#sentiment scoring
for i in range(0, len(grouped_texts), batch_size):
    batch = grouped_texts[i:i+batch_size]
    try:
        results = sentiment_model(batch)
        batch_scores = [r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results]
        sentiments.extend(batch_scores)
    except Exception as e:
        print(f"Sentiment batch error at index {i}: {e}")
        sentiments.extend([0.0] * len(batch))  # fallback neutral sentiment

df_analysis = pd.DataFrame({
    "document": grouped_texts,
    "topic": topics,
    "sentiment": sentiments
})

topic_sentiment = df_analysis.groupby("topic").sentiment.agg(["mean", "count"]).reset_index()
topic_sentiment.columns = ["topic", "avg_sentiment", "doc_count"]


df_analysis.to_csv("discord_topic_assignments_with_sentimentgen-event.csv", index=False)
topic_sentiment.to_csv("discord_topic_sentiment_summarygen-event.csv", index=False)
print('Done with CSVs!')


filtered = topic_sentiment[topic_sentiment["topic"] != -1]  # exclude outliers

plt.figure(figsize=(12, 6))
plt.bar(filtered["topic"].astype(str), filtered["avg_sentiment"], color="skyblue")
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Topic ID")
plt.ylabel("Average Sentiment (Transformer)")
plt.title("Average Sentiment by Topic")
plt.tight_layout()
plt.savefig("sentiment_by_topic.png")
plt.show()
