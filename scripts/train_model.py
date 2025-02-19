import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import pandas as pd
from datasets import Dataset

# Load the WER Analysis CSV
df = pd.read_csv("/Users/dharshusivakumar/Desktop/ATC_Project/atc-dataset/WER_Analysis.csv")

# Remove empty transcriptions (WER = 1.0 and no words transcribed)
df = df.dropna(subset=["transcribed_text", "text"])

# Create input-output pairs for training
df["input_text"] = "Fix: " + df["transcribed_text"]
df["target_text"] = df["text"]

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(df[["input_text", "target_text"]])

# Split into train/test sets (90% train, 10% test)
dataset = dataset.train_test_split(test_size=0.1)

# Load T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Tokenization function
def tokenize_data(example):
    return tokenizer(
        example["input_text"], padding="max_length", truncation=True, max_length=128
    )

# Tokenize dataset
dataset = dataset.map(tokenize_data, batched=True)

# Convert dataset to TensorFlow format
train_dataset = dataset["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["target_text"],
    batch_size=8,
    shuffle=True
)

test_dataset = dataset["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["target_text"],
    batch_size=8,
    shuffle=False
)

# Load pre-trained T5 model for TF
model = TFT5ForConditionalGeneration.from_pretrained("t5-small")

# Compile model with Adam optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Train the model using TensorFlow
model.fit(train_dataset, validation_data=test_dataset, epochs=3)

# Save the model
model.save_pretrained("./t5_error_correction")
tokenizer.save_pretrained("./t5_error_correction")

print("Model training complete and saved!")
