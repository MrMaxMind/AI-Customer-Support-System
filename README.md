# AI Customer Support System

Welcome to the AI Customer Support System repository. This project utilizes a fine-tuned GPT-2 model with LoRa adapters to provide automated responses to customer queries. It features a web-based interface allowing users to input their queries and receive AI-generated responses in real time. Below is an overview of the project, its features, and instructions for running and deploying the app.

---

<div align="center">
  <img src="./AI_image.png" alt="AI Customer Support System" style="border:none;">
</div>

## Overview

This project leverages a GPT-2 model fine-tuned with LoRa adapters to assist in customer support automation. It allows users to input queries related to products or services, and the system generates relevant responses based on the input. The project features a Flask-based backend, an interactive Bootstrap frontend, and model deployment through Render using Docker.

---

## Features

- **LoRa Fine-Tuning**: GPT-2 model is fine-tuned using Low-Rank Adaptation (LoRa) for better customization and performance.
- **Interactive Web Interface**: A user-friendly web interface built with Bootstrap, allowing real-time user interaction.
- **Flask Backend**: Handles model loading, query processing, and response generation.
- **GPU Support**: CUDA-based inference to accelerate response generation.
- **Real-Time Predictions**: Provides fast, AI-generated responses to customer queries.

---

## Contents

- **app.py**: Flask backend that handles the model loading, input processing, and prediction API.
- **models/** : Contains the fine-tuned GPT-2 model with LoRa adapters.
- **templates/index.html**: Frontend HTML for user input and response display.
- **static/style.css**: Custom styling for the web interface.
- **static/app.js**: JavaScript for handling frontend interactions and AJAX requests.
- **Dockerfile**: Docker configuration for deploying the app on Render.
- **requirements.txt**: Lists the required dependencies to run the project.

---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/AI-Customer-Support-System.git
   cd AI-Based-Customer-Support-System
2. **Install the required packages**:
   bash
   pip install -r requirements.txt
   
3. **Run the Jupyter Notebook**:
   ```bash
   python3 app.py

4. **Access the Web App**: Open your web browser and go to `http://127.0.0.1:5000/`

---

## Deployment Instructions

### 1. Model Fine-Tuning and Preparation: 

Fine-tune the **GPT-2** model using the LoRa method and save the model files (`adapter_model.safetensors`, `adapter_config.json`) in the `models/` directory.

### 2. Docker Setup and Deployment on Render:

- Ensure the `Dockerfile` is correctly set up.
- Push your project to GitHub:
  ```bash
  git push origin main
- Create a new web service on Render, connecting it to your GitHub repository.
- Ensure `app.py` is set as the entry point, and all dependencies from `requirements.txt` are installed.

### 3. Configure the Environment:

- Add the necessary environment variables if required.
- Deploy the app, and Render will automatically start your service.

---

## Key Insights

- Successfully implemented LoRa fine-tuning for a GPT-2 model to handle customer queries.
- Deployed a fast and efficient customer support system using Docker and Render.
- Integrated a user-friendly web interface with a Flask backend for real-time predictions

---

## Tools and Libraries

- `Flask`: For the web backend and API routing.
- `Hugging Face Transformers`: For model loading and text generation.
- `PEFT (Parameter-Efficient Fine-Tuning)`: For LoRa fine-tuning.
- `CUDA`: For GPU-based inference acceleration.
-`Bootstrap`: For frontend design.
-`HTML, CSS, JavaScript`: For creating the user interface.

---

## Contributing
If you have suggestions or improvements, feel free to open an issue or create a pull request.

---

## *Thank you for visiting! If you find this project useful, please consider starring the repository. Happy coding!*
