# Piperag GGML

Piperag GGML is a Python-based service that combines high-performance language model inference with image recognition capabilities. Originally built around GGML for efficient inference, the project has evolved to include dynamic chat service support using the MLCEngine (MLC LLM) and image recognition features for tasks such as deepfake detection.

## Overview

Piperag GGML offers a modular framework for:
- **Chat Service:** Interact with language models using an MLCEngine wrapper that supports models hosted on Hugging Face. The chat service is dynamically reloadable and adjustable for multiple model types.
- **Image Recognition:** Process images for various tasks (e.g., deepfake detection) via a dedicated endpoint.
- **Dynamic Model Updates:** Easily switch or update models (both chat and image recognition) at runtime using REST API endpoints.

## Key Features

- **MLC LLM Integration:**  
  - Leverages [MLC LLM](https://huggingface.co/eccsm/mlc_llm) for language model inference.
  - Uses a custom wrapper with adjustments (e.g., Pydantic private attributes) to avoid field errors.
  - Supports direct Hugging Face links—ensure your model repository includes all required artifacts (like `ndarray-cache.json`).

- **Image Recognition Service:**  
  - Provides endpoints for image recognition tasks.
  - Easily extendable to support models for deepfake detection or other computer vision tasks.
  
- **Dynamic Reloading:**  
  - Update models on the fly using the `/update_model` endpoint.
  - Seamlessly switch between different model types (e.g., `mlc_llm` vs. `vicuna_ggml`).

- **REST API Endpoints:**  
  - `/ask` – for chat queries.
  - `/update_model` – for dynamically reloading/updating the model configuration.
  - `/recognize` – for performing image recognition tasks.

## Getting Started

### Prerequisites

- **Python 3.10+** (recommended for best compatibility)
- [GGML](https://github.com/ggerganov/ggml) libraries and build tools (if you plan to modify or extend low-level inference)
- [CMake](https://cmake.org) and a C/C++ compiler (if building native extensions or GGML components)
- [Git LFS](https://git-lfs.github.com) for managing large model files

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/eccsm/piperag_ggml.git
   cd piperag_ggml
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   If you have a `requirements.txt` (or use [Poetry/Pipenv] if preferred), install with:

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Build Docker Image:**

   The provided `Dockerfile` can be used to containerize the application:

   ```bash
   docker build -t piperag_ggml .
   docker run -p 8000:8000 piperag_ggml
   ```

## Usage

### Running the Application

You can start the service using Uvicorn (or via Docker):

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### API Endpoints

- **Chat Service:**  
  **GET** `/ask`  
  **Query Parameters:**
  - `q` (string): The user query.
  - `model` (optional string): Specify the model type (e.g., `mlc_llm` or `vicuna_ggml`).

  **Example:**

  ```
  GET http://127.0.0.1:8000/ask?q=Hello+world&model=mlc_llm
  ```

- **Dynamic Model Update:**  
  **POST** `/update_model`  
  **Body Parameters (JSON):**
  - `new_model` (string): New model file or HF path.
  - `new_model_type` (optional string): Specify the new model type.
  
  **Example Payload:**

  ```json
  {
    "new_model": "HF://eccsm/mlc_llm/RedPajama-INCITE-Chat-3B-v1-q4f16_1",
    "new_model_type": "mlc_llm"
  }
  ```

- **Image Recognition:**  
  **POST** `/recognize`  
  **Query Parameters:**
  - `task` (string): Specify the recognition task (e.g., "deepfake_detection").
  
  **File Upload:**
  - `image` (file): Upload the image file.

  **Example:**

  Use a tool like Postman or a cURL command to upload an image file along with the task parameter.

## Model Configuration & Hugging Face Integration

- **MLC LLM Models:**  
  The MLCEngine wrapper requires a Hugging Face model repository that contains:
  - The compiled shared library (e.g., `RedPajama-INCITE-Chat-3B-v1-q4f16_1-vulkan.so`)
  - Additional model files such as `ndarray-cache.json` and parameter shards.

  **Important:**  
  Make sure your HF repository (e.g., [eccsm/mlc_llm](https://huggingface.co/eccsm/mlc_llm)) has the full directory structure required by MLC LLM.

- **Image Recognition Models:**  
  Similarly, if you integrate or update image recognition models, ensure that their paths (local or on HF) are correctly set in your configuration.

## Contributing

Contributions are welcome! Feel free to fork the repository, submit pull requests, or open issues to discuss improvements and bug fixes.

## License

This project is licensed under the MIT License.

## Contact

For questions or further information, please open an issue on GitHub or contact Ekincan Casim via LinkedIn.

