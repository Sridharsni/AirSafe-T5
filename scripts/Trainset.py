import pandas as pd

# Load the datasets
test_file = "/Users/dharshusivakumar/Desktop/ATC_Project/atc-dataset/data/test-00000-of-00001.parquet"
train_file_1 = "/Users/dharshusivakumar/Desktop/ATC_Project/atc-dataset/data/train-00000-of-00002.parquet"
train_file_2 = "/Users/dharshusivakumar/Desktop/ATC_Project/atc-dataset/data/train-00001-of-00002.parquet"

test_df = pd.read_parquet(test_file)
train_df_1 = pd.read_parquet(train_file_1)
train_df_2 = pd.read_parquet(train_file_2)

# Combine all datasets
df = pd.concat([test_df, train_df_1, train_df_2])

# Keep only the text column
df = df[['text']]

# Convert text to lowercase for consistency
df['text'] = df['text'].str.lower()

# Remove special characters and numbers
df['text'] = df['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

# Display cleaned data
print("\nCleaned Text Data Sample:")
print(df.head())

# Save as CSV for easier exploration
df.to_csv("cleaned_atc_text.csv", index=False)
