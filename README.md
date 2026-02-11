# Trader-Performance-Vs-Market-Sentiment

# Project Overview

This project analyzes how **market sentiment (Fear / Greed)** influences:

* Trader performance (PnL, win rate)
* Trading behavior (frequency, risk level, position size)
* Strategy outcomes

Using real datasets:

* Fear & Greed Index
* Hyperliquid trading activity

We perform:

1️⃣ Data preparation
2️⃣ Data analysis (EDA)
3️⃣ Behavioral segmentation
4️⃣ Sentiment–performance insights
5️⃣ Profitability prediction using ML

This project connects **behavioral finance + data science + trading analytics**.



# Problem Statement

Markets are heavily driven by psychology.

* Fear → panic selling
* Greed → risky overtrading

Emotions often cause unstable performance and excessive risk-taking. ([one-signal.com][2])

This project answers:

* Does trader performance change during Fear vs Greed?
* Do traders behave differently in different sentiment phases?
* Can we predict profitability using sentiment + behavior?


#  Datasets Used

## 1 Fear & Greed Index (`fear_greed_index.csv`)

Represents overall crypto market sentiment.

Columns:

* `date`
* `value` (0–100 sentiment score)
* `classification`

Sentiment scale:

* 0–24 → Extreme Fear
* 25–49 → Fear
* 50–74 → Neutral/Greed
* 75–100 → Extreme Greed

This dataset captures the **emotional state of the market**.

---

## 2️ Hyperliquid Trading Data (`historical_data.csv`)

Contains real trader-level activity.

Key columns:

* `Account`
* `Coin`
* `Side`
* `Execution Price`
* `Size USD`
* `Closed PnL`
* `Timestamp`

This dataset shows:

* What traders did
* When they traded
* How much they earned/lost


#  Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib / Seaborn
* Scikit-learn


# PART 1 — Data Preparation

## Step 1 — Load datasets

```python
fear = pd.read_csv('fear_greed_index.csv')
trades = pd.read_csv('historical_data.csv')
```

## Step 2 — Basic inspection

* Rows & columns
* Column names
* First few rows

## Step 3 — Clean data

* Remove duplicates
* Handle missing values
* Convert timestamps

```python
fear['date'] = pd.to_datetime(fear['date'])
trades['Timestamp'] = pd.to_datetime(trades['Timestamp'])
```

## Step 4 — Align by day

```python
fear['day'] = fear['date'].dt.date
trades['day'] = trades['Timestamp'].dt.date

merged = pd.merge(trades, fear, on='day', how='left')
```

Now each trade is tagged with:

* Sentiment value
* Sentiment classification


#  PART 2 — Feature Engineering

Created key metrics:

## Trader Performance

* Daily PnL per trader
* Win/Loss indicator
* Win rate

```python
merged['Win'] = merged['Closed PnL'] > 0
```

## Trading Behavior

* Trade size
* Trades per day
* Long vs Short ratio

## Risk Metric

* Risk_Level (Low / Medium / High leverage proxy)

---

#  PART 3 — Exploratory Data Analysis

## Performance vs Sentiment

Analyzed:

* Average PnL by sentiment
* Profit vs loss distribution
* Win rate by sentiment

Key finding types:

* Fear days often show different PnL patterns
* Greed days often show higher trading activity


## Behavior vs Sentiment

Analyzed:

* Trades per day
* Average trade size
* Risk level distribution
* Long/Short bias

Psychology research suggests:

* Fear → hesitation, panic exits
* Greed → aggressive risk-taking 


#  PART 4 — Behavioral Segmentation

## SEGMENT 1 — High vs Low Leverage Traders

Using:

```python
Risk_Level
```

Compared:

* Avg PnL
* Win rate
* Trade frequency

---

## SEGMENT 2 — Frequent vs Infrequent Traders

Grouped by:

* Trades per day per account

Compared:

* Profitability
* Consistency

---

## SEGMENT 3 — Consistent vs Inconsistent Traders

Based on:

* Win rate
* PnL stability

---

#  PART 5 — Key Insights

Examples of insights extracted:

* Traders take bigger risks during Greed
* Trading activity spikes during emotional extremes
* Profitability patterns differ across sentiment phases

Sentiment has been shown in research to influence returns and market behavior. 


# 🤖 PART 6 — Machine Learning Model

## Goal

Predict whether a trader will be:

* Profitable
* Not profitable

## Target Variable

```python
daily['Profit_Bucket'] = (daily['Daily_PnL'] > 0).astype(int)
```

## Features Used

* Avg trade size
* Trade count
* Risk level
* Sentiment classification

## Encoding

```python
daily = pd.get_dummies(daily, columns=['classification','Risk_Level'], drop_first=True)
```

## Model

```python
RandomForestClassifier
```

## Evaluation

* Accuracy
* Classification report

---

#  Final Output

This system provides:

* Behavioral insights
* Performance analysis
* Strategy timing signals
* Profitability prediction

---

#  Project Structure

```
├── fear_greed_index.csv
├── historical_data.csv
├── merged_data.csv
├── notebook.ipynb
├── README.md
```

---

#  How to Run

1️. Open Jupyter / Colab
2️. Install libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

3️. Run notebook cells step by step:

* Data loading
* Cleaning
* EDA
* Feature engineering
* ML model


