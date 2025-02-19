from transformers import T5ForConditionalGeneration, T5Tokenizer

def run_inference(model_path, input_text):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output_ids = model.generate(input_ids)
    corrected_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"Corrected Text: {corrected_text}")
    return corrected_text

if __name__ == "__main__":
    input_text = "enter transcribed text needing correction"
    run_inference("models/t5_atc_corrector", input_text)
