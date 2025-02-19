import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

def test_model(model_path, test_csv):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    df = pd.read_csv(test_csv)
    
    df['clean_text'] = df['clean_text'].astype(str)
    predictions = []
    
    for text in df['clean_text']:
        input_ids = tokenizer.encode(text, return_tensors='pt')
        output_ids = model.generate(input_ids)
        predictions.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    
    df['predicted_text'] = predictions
    df.to_csv("test_results.csv", index=False)
    print("Test results saved to test_results.csv")

if __name__ == "__main__":
    test_model("models/t5_atc_corrector", "data/test_atc_text.csv")
