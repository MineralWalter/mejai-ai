from ollama import chat


MODEL_NAME = "qwen3:4b"


def ask_model(prompt: str) -> str:
    """
    Send a prompt to the local Ollama model
    and return its text response.
    """

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.message.content