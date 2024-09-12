from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

app = Flask(__name__)

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
peft_model = PeftModel.from_pretrained(model, "./models").to(device)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.json.get('user_input')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    # Tokenize input and move to device (GPU if available)
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    
    # Generate output from the model
    outputs = peft_model.generate(
        inputs.input_ids, 
        max_length=500, 
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated output, skipping the input text (query)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(user_input, "").strip()
    
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
