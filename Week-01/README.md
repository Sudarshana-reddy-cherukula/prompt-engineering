# Week 01 — Environment Setup for Prompt Engineering

## 1. Aim
To set up the Python environment required for prompt engineering, configure access to a Large Language Model (LLM), make the first API call, and verify successful communication with the model.

## 2. Objective
- Understand the basic requirements of prompt engineering.
- Explore different ways to access an LLM.
- Install Python and the required libraries.
- Configure API credentials securely.
- Execute a simple prompt and display the response.
- Identify common API errors.

## 3. Introduction
Prompt engineering is the process of designing and refining instructions given to a Large Language Model (LLM) to obtain useful and relevant responses.

Prompt engineering requires access to an LLM, such as GPT-4, GPT-3.5, Claude, or Llama.

## 4. Methods of Accessing an LLM

### 4.1 OpenAI API
- Requires an API key.
- Uses the `openai` Python package.
- Provides access to supported OpenAI models.

### 4.2 Cohere API
- An alternative for accessing language models.
- Requires appropriate API credentials and the relevant Python package.

### 4.3 Local Models
- Models can be run locally using tools such as Ollama and Hugging Face.
- Local execution may not require a hosted API key.
- Requires suitable local computing resources.

## 5. Software Requirements
- Python 3.8 or later, as specified in the experiment notes.
- Python package manager (pip).
- An LLM provider or local model runtime.
- An internet connection for hosted API access.
- A code editor or terminal.

## 6. Environment Setup

### Step 1: Verify Python Installation

Run the following command:

```bash
python --version