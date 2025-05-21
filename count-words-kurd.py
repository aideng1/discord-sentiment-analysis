#this script analyses incidence of term usage and compares it against the total
#message volume. Dumps data to a CSV and provides graphic chart with trendlines
import pandas as pd
import glob
import re
from pathlib import Path
import matplotlib.pyplot as plt

#settings to gather incidence
csv_path = "general_all/*.csv" #location of CSVs, wildcard grabs all within folder
text_column = "content" #message content to be analysed
timestamp_column = "date" #optional timestamping for batching by day/time
terms = ["daleel","jash","rifa","dersim","slemani"] #terms to look for

#regex
pattern = re.compile(r"\b(" + "|".join(terms) + r")\b", flags=re.IGNORECASE)

#script
all_data = []

for file in glob.glob(csv_path):
    try:
        df = pd.read_csv(file)
        if text_column not in df.columns or timestamp_column not in df.columns:
            print(f"Skipping {file}: required columns missing")
            continue

        df = df[[timestamp_column, text_column]].dropna()
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
        df = df.dropna(subset=[timestamp_column])
        df["mention"] = df[text_column].str.contains(pattern)
        all_data.append(df)

    except Exception as e:
        print(f"Error processing {file}: {e}")

#aggregate data
if all_data:
    combined = pd.concat(all_data)
    combined["date"] = combined[timestamp_column].dt.date

    daily_counts = (
        combined
        .groupby("date")
        .agg(total_messages=(text_column, "count"), kurd_mentions=("mention", "sum"))
        .reset_index()
    )

    daily_counts["percent"] = 100 * daily_counts["kurd_mentions"] / daily_counts["total_messages"]
    daily_counts["percent_ma7"] = daily_counts["percent"].rolling(window=7, min_periods=1).mean()
    daily_counts["volume_ma7"] = daily_counts["total_messages"].rolling(window=7, min_periods=1).mean()

    max_percent = daily_counts["percent_ma7"].max()
    max_volume = daily_counts["volume_ma7"].max()
    scale_factor = max_percent / max_volume if max_volume > 0 else 1

    daily_counts["scaled_volume_ma7"] = daily_counts["volume_ma7"] * scale_factor
    daily_counts["trend_delta"] = daily_counts["percent_ma7"] - daily_counts["scaled_volume_ma7"]

    #dump to csv
    daily_counts.to_csv("SomeTitle.csv", index=False)

    #visually depict data
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(daily_counts["date"], daily_counts["percent"], color="blue", label="Daily % Mention", linewidth=1.0)
    ax1.plot(daily_counts["date"], daily_counts["percent_ma7"], color="navy", linestyle="--", linewidth=2, label="7-Day Avg %")
    ax1.set_ylabel("Mentions of keywords (%)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True)
    ax1.plot(daily_counts["date"], daily_counts["scaled_volume_ma7"], color="gray", linestyle="--", linewidth=2, label="7-Day Avg Volume (scaled)")

    ax2 = ax1.twinx()
    ax2.bar(daily_counts["date"], daily_counts["total_messages"], alpha=0.3, color="lightgray", label="Daily Volume")
    ax2.set_ylabel("Total Messages", color="gray")
    ax2.tick_params(axis='y', labelcolor="gray")

    fig.suptitle("Mentions of keywords': Trend vs. Total Volume", fontsize=14)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("kurd_mentions_with_dual_trend_and_delta.png")
    plt.show()

    print("\n📉 Delta Summary (percent trend - scaled volume trend):")
    print(daily_counts[["date", "trend_delta"]].describe())
