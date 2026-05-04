# %% [markdown]
# # Section B: Data Understanding and EDA
# 
# Mobile money fraud detection — PaySim dataset
# Author: Eyram
# Date: April 2026





# %%
# Standard imports for data analysis
import pandas as pd          # tabular data handling
import numpy as np           # numerical operations
import matplotlib.pyplot as plt   # plotting
import seaborn as sns        # statistical visualisation built on matplotlib

# Display settings: show more columns when printing dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

# Plot styling
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)


# %%
# Load the raw dataset
# Path is relative to the project root, since we'll run from there
df = pd.read_csv('/home/wheezy20/projects/mobile_money_fraud/data/raw/paysim.csv')

print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# %%
# Inspect column data types
print("Column types:")
print(df.dtypes)
# %%

# %%
# Section B Q2: What is the actual fraud rate?

# Count of each class (0 = legitimate, 1 = fraud)
class_counts = df['isFraud'].value_counts()
print("Class counts:")
print(class_counts)
print()

# As a percentage
class_pct = df['isFraud'].value_counts(normalize=True) * 100
print("Class proportions (%):")
print(class_pct.round(4))
print()

# Confirm the imbalance ratio
n_legit = class_counts[0]
n_fraud = class_counts[1]
ratio = n_legit / n_fraud
print(f"Imbalance ratio: 1 fraudulent transaction for every {ratio:.0f} legitimate ones")
# %%
# %%
# Section B Q3: Are there any missing values?

# Count missing values per column
missing = df.isnull().sum()
print("Missing values per column:")
print(missing)
print()

# Total missing
total_missing = missing.sum()
print(f"Total missing values across all columns: {total_missing}")

# Quick numeric summary — gives us min, max, mean for each numeric column
# This will help us spot weird values like negative balances
print("\nDescriptive statistics for numeric columns:")
print(df.describe().round(2))
# %%
# %%
# Section B Q4: Compare fraud vs legitimate transactions

# Group by isFraud and look at numeric column means
print("Mean values by class (0 = legitimate, 1 = fraud):")
print(df.groupby('isFraud')[['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                              'oldbalanceDest', 'newbalanceDest']].mean().round(2))
print()

# How often is isFlaggedFraud actually triggered?
print("isFlaggedFraud counts:")
print(df['isFlaggedFraud'].value_counts())
print()

# When isFlaggedFraud=1, is it always also isFraud=1?
flagged_fraud_overlap = df[df['isFlaggedFraud'] == 1]['isFraud'].value_counts()
print("When isFlaggedFraud=1, what is isFraud?")
print(flagged_fraud_overlap)
# %%
# %%
# Section B Q5: Where does fraud actually occur? Break down by transaction type

# Total counts and fraud counts per transaction type
type_summary = df.groupby('type').agg(
    total_count=('isFraud', 'size'),
    fraud_count=('isFraud', 'sum')
)
type_summary['fraud_rate_pct'] = (type_summary['fraud_count'] / type_summary['total_count'] * 100).round(4)
type_summary['pct_of_all_fraud'] = (type_summary['fraud_count'] / type_summary['fraud_count'].sum() * 100).round(2)

print("Transaction breakdown by type:")
print(type_summary.sort_values('fraud_count', ascending=False))
# %%
# %%
# Section B Q6: Visualise the fraud distribution by transaction type

import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data for plotting (re-using the type_summary table)
type_summary_sorted = type_summary.sort_values('fraud_count', ascending=False)

# Create a figure with two side-by-side panels
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Total transaction count by type (log scale for visibility)
sns.barplot(
    x=type_summary_sorted.index,
    y=type_summary_sorted['total_count'],
    ax=axes[0],
    color='steelblue'
)
axes[0].set_title('Total Transactions by Type')
axes[0].set_ylabel('Number of transactions')
axes[0].set_xlabel('Transaction type')
axes[0].tick_params(axis='x', rotation=30)

# Panel 2: Fraud rate per type (as a percentage)
sns.barplot(
    x=type_summary_sorted.index,
    y=type_summary_sorted['fraud_rate_pct'],
    ax=axes[1],
    color='crimson'
)
axes[1].set_title('Fraud Rate by Transaction Type (%)')
axes[1].set_ylabel('Fraud rate (%)')
axes[1].set_xlabel('Transaction type')
axes[1].tick_params(axis='x', rotation=30)

# Add value labels on top of each bar in panel 2
for i, val in enumerate(type_summary_sorted['fraud_rate_pct']):
    axes[1].text(i, val + 0.02, f'{val:.2f}%', ha='center', fontsize=9)

plt.tight_layout()

# Save the figure for use in your report
plt.savefig('reports/figures/fig_01_fraud_by_type.png', dpi=150, bbox_inches='tight')
plt.show()

print("Figure saved to reports/figures/fig_01_fraud_by_type.png")
# %%# %%
# Section B Q7: Compare transaction amount distributions for fraud vs legitimate
# Restrict to CASH_OUT and TRANSFER only — where fraud actually exists

# Create a filtered subset: only the transaction types where fraud occurs
df_fraud_types = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()

# Drop zero-amount transactions for plotting (log scale needs positive values)
plot_data = df_fraud_types[df_fraud_types['amount'] > 0]

print(f"Filtered dataset shape: {df_fraud_types.shape}")
print(f"Fraud rate in filtered set: {df_fraud_types['isFraud'].mean() * 100:.4f}%")
print(f"Zero-amount transactions excluded from plot: {(df_fraud_types['amount'] == 0).sum()}")
print()

# Plot: amount distributions on log scale (because of heavy skew)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: histogram with log-spaced bins for proper shape
import numpy as np
log_bins = np.logspace(
    np.log10(plot_data['amount'].min()),
    np.log10(plot_data['amount'].max()),
    80
)

axes[0].hist(
    plot_data[plot_data['isFraud'] == 0]['amount'],
    bins=log_bins, alpha=0.6, label='Legitimate', color='steelblue', density=True
)
axes[0].hist(
    plot_data[plot_data['isFraud'] == 1]['amount'],
    bins=log_bins, alpha=0.6, label='Fraud', color='crimson', density=True
)
axes[0].set_xscale('log')
axes[0].set_xlabel('Transaction amount (log scale)')
axes[0].set_ylabel('Density')
axes[0].set_title('Amount distribution: fraud vs legitimate')
axes[0].legend()

# Panel 2: boxplot using hue parameter (newer seaborn syntax)
sns.boxplot(
    data=plot_data,
    x='isFraud',
    y='amount',
    hue='isFraud',
    palette={0: 'steelblue', 1: 'crimson'},
    legend=False,
    ax=axes[1]
)
axes[1].set_yscale('log')
axes[1].set_xlabel('isFraud (0=legitimate, 1=fraud)')
axes[1].set_ylabel('Transaction amount (log scale)')
axes[1].set_title('Amount boxplot by class')
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Legitimate', 'Fraud'])

plt.tight_layout()
plt.savefig('reports/figures/fig_02_amount_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("Figure saved to reports/figures/fig_02_amount_distribution.png")
# %%
# %%
# Diagnostic: what's happening with the amount distribution?

# Look at amount stats split by class
print("Amount statistics by class:")
print(df_fraud_types.groupby('isFraud')['amount'].describe().round(2))
print()

# How many tiny transactions are there?
tiny = df_fraud_types[df_fraud_types['amount'] < 1]
print(f"Transactions with amount < 1: {len(tiny)}")
print(f"  How many are fraud: {tiny['isFraud'].sum()}")
print()

# What about really small (under 100)?
small = df_fraud_types[df_fraud_types['amount'] < 100]
print(f"Transactions with amount < 100: {len(small)}")
print(f"  How many are fraud: {small['isFraud'].sum()}")
# %%
# %%
# Save the filtered dataset for use in later sections
# This is a checkpoint — once saved, we can load it directly next time
df_fraud_types.to_csv('data/processed/paysim_filtered.csv', index=False)
print(f"Saved filtered dataset: {df_fraud_types.shape}")
print("Path: data/processed/paysim_filtered.csv")
# %%
# %% [markdown]
# ## Decision log: visualisation choices
#
# **Issue identified:** The first attempt at visualising the transaction-amount
# distribution (histogram, fig_02 panel 1) produced a misleading result.
# The fraud (red) histogram appeared centred around 100–1,000, while the
# accompanying boxplot (panel 2) clearly placed the fraud median at ~441,000.
# The two views of the same data disagreed.
#
# **Diagnostic:** Investigation showed:
# - 19 transactions had amount < 1, of which 16 were fraud.
# - 801 transactions had amount < 100, of which only 18 were fraud (the rest legitimate).
# - The remaining tail of tiny legitimate transactions, combined with density
#   normalisation against vastly different sample sizes (2.76M legit vs 8.2K fraud),
#   visually compressed the legitimate distribution and inflated the apparent
#   density of fraud at low amounts.
#
# **Conclusion:** The boxplot (panel 2) is the truthful representation.
# It correctly shows that fraud transactions have a higher median amount
# (~441K vs ~171K for legitimate) and a tighter, higher distribution overall.
# The histogram was an artefact of density normalisation, not a property of
# the data.
#
# **Decision:** The boxplot will be used as the primary amount-distribution
# figure in the report. The histogram is retained in the script for
# methodological transparency but will not be cited in the final write-up.
#
# ## Decision log: data scope for modelling
#
# - Modelling will use `df_fraud_types` — i.e. only CASH_OUT and TRANSFER
#   transactions (~2.77M rows). The other three transaction types contain
#   zero fraud cases and would only add noise.
# - The 16 zero-amount transactions are retained, despite being unusual,
#   because all 16 are confirmed fraud and may carry signal.
# - The filtered dataset is saved to `data/processed/paysim_filtered.csv`
#   for reproducible use in subsequent sections.
# %%



# %%
# Reload the filtered dataset from disk
# This skips the full 6.3M-row load — much faster
import pandas as pd
from pathlib import Path

DATA_PATH = Path.home() / 'projects' / 'mobile_money_fraud' / 'data' / 'processed' / 'paysim_filtered.csv'

print(f"Loading from: {DATA_PATH}")
print(f"File exists: {DATA_PATH.exists()}")

df_fraud_types = pd.read_csv(DATA_PATH)

print(f"Filtered dataset shape: {df_fraud_types.shape}")
print(f"Fraud rate: {df_fraud_types['isFraud'].mean() * 100:.4f}%")








# %%
# Section B Q8: Verify balance patterns and engineer initial signal features
# Working from the filtered dataset (CASH_OUT and TRANSFER only)

# Compute balance change for sender and recipient
df_fraud_types['balance_change_orig'] = (
    df_fraud_types['oldbalanceOrg'] - df_fraud_types['newbalanceOrig']
)
df_fraud_types['balance_change_dest'] = (
    df_fraud_types['newbalanceDest'] - df_fraud_types['oldbalanceDest']
)

# Compute the balance equation error:
# For a legitimate transfer, sender's balance should drop by exactly `amount`,
# and recipient's balance should rise by exactly `amount`.
# Any deviation is suspicious.
df_fraud_types['error_orig'] = (
    df_fraud_types['amount'] - df_fraud_types['balance_change_orig']
)
df_fraud_types['error_dest'] = (
    df_fraud_types['amount'] - df_fraud_types['balance_change_dest']
)

# Compute the drainage ratio (what fraction of sender's balance was sent)
# Avoid division by zero by adding a tiny constant
df_fraud_types['drainage_ratio'] = (
    df_fraud_types['amount'] / (df_fraud_types['oldbalanceOrg'] + 1)
)

# Show the means of these new columns, split by fraud class
print("New balance-related features, mean by class:")
print(df_fraud_types.groupby('isFraud')[
    ['balance_change_orig', 'balance_change_dest', 'error_orig', 'error_dest', 'drainage_ratio']
].mean().round(2))
print()

# Also show medians, which are more robust to outliers
print("Same features, median by class:")
print(df_fraud_types.groupby('isFraud')[
    ['balance_change_orig', 'balance_change_dest', 'error_orig', 'error_dest', 'drainage_ratio']
].median().round(2))

# %%


# %%
# Section B Q9: Visualise the drainage pattern that distinguishes fraud
# Drainage ratio close to 1.0 means the entire sender balance was transferred

# We need to handle two issues for plotting:
# 1. Many transactions have oldbalanceOrg = 0, which gives a meaningless ratio
# 2. The ratio can be very large (>1) when amount > balance, which happens in PaySim
# So we filter to senders with a meaningful starting balance and clip the ratio at 2

plot_drain = df_fraud_types[df_fraud_types['oldbalanceOrg'] > 0].copy()
plot_drain['drainage_clipped'] = plot_drain['drainage_ratio'].clip(upper=2)

print(f"Transactions with oldbalanceOrg > 0: {len(plot_drain):,}")
print(f"  Of which fraud: {plot_drain['isFraud'].sum():,}")
print()

# Plot: density of drainage ratio, fraud vs legitimate
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
    plot_drain[plot_drain['isFraud'] == 0]['drainage_clipped'],
    bins=50, alpha=0.6, label='Legitimate', color='steelblue', density=True
)
ax.hist(
    plot_drain[plot_drain['isFraud'] == 1]['drainage_clipped'],
    bins=50, alpha=0.6, label='Fraud', color='crimson', density=True
)

# Mark the 1.0 line — the "full drain" signature
ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
ax.text(1.02, ax.get_ylim()[1] * 0.9, 'Full drain (ratio = 1.0)',
        fontsize=9, color='black')

ax.set_xlabel('Drainage ratio (amount / oldbalanceOrg, clipped at 2)')
ax.set_ylabel('Density')
ax.set_title('Fraud transactions cluster at drainage ratio = 1.0 (full drain)')
ax.legend()

plt.tight_layout()
plt.savefig('reports/figures/fig_03_drainage_pattern.png', dpi=150, bbox_inches='tight')
plt.show()

print("Figure saved to reports/figures/fig_03_drainage_pattern.png")
# %%



# %%
# Section B Q10: Time patterns and identity columns

# Time-of-day analysis
# step is hours; step % 24 gives hour-of-day
df_fraud_types['hour_of_day'] = df_fraud_types['step'] % 24

print("Hour-of-day fraud rate (top 5 highest):")
hourly = df_fraud_types.groupby('hour_of_day').agg(
    total=('isFraud', 'size'),
    fraud_rate_pct=('isFraud', lambda x: x.mean() * 100)
)
print(hourly.sort_values('fraud_rate_pct', ascending=False).head())
print()

# Identity columns: how many unique senders/recipients?
print(f"Unique senders (nameOrig):    {df_fraud_types['nameOrig'].nunique():,}")
print(f"Unique recipients (nameDest): {df_fraud_types['nameDest'].nunique():,}")
print(f"Total transactions:           {len(df_fraud_types):,}")
print()

# Are recipients reused often? (mule accounts would show this)
recipient_counts = df_fraud_types['nameDest'].value_counts()
print("Top 5 most-frequent recipients:")
print(recipient_counts.head())
print()
print("Of these top recipients, what fraction received fraud?")
top_recipients = recipient_counts.head(5).index
for r in top_recipients:
    subset = df_fraud_types[df_fraud_types['nameDest'] == r]
    print(f"  {r}: {len(subset)} txns, {subset['isFraud'].sum()} fraud "
          f"({subset['isFraud'].mean() * 100:.2f}%)")
# %%





# %%
# Save the dataset with engineered features for use in Section C
df_fraud_types.to_csv('data/processed/paysim_filtered_with_features.csv', index=False)
print(f"Saved: data/processed/paysim_filtered_with_features.csv")
print(f"Shape: {df_fraud_types.shape}")
# %%
