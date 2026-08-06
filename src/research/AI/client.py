from ollama import Client, ResponseError

MODEL_NAME = "qwen3:4b-instruct"
OLLAMA_HOST = "http://localhost:11434"

REQUEST_TIMEOUT_SECONDS = 300.0
MAX_RESPONSE_TOKENS = 2_400


client = Client(
    host=OLLAMA_HOST,
    timeout=REQUEST_TIMEOUT_SECONDS,
)


def ask_model(prompt: str) -> str:
    """
    Generate only the final grounded report using Qwen3's
    non-thinking instruct model.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The Ollama prompt must be a non-empty string")

    print(
        f"[AI] Generating final report with {MODEL_NAME}...",
        flush=True,
    )

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only the finished Markdown report requested "
                        "by the user. Do not describe your planning, reasoning, "
                        "drafting process, or instructions."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt.strip(),
                },
            ],
            think=False,
            stream=False,
            options={
                "temperature": 0.1,
                "num_predict": MAX_RESPONSE_TOKENS,
            },
            keep_alive="5m",
        )
    except ResponseError as error:
        status = (
            f" (status {error.status_code})"
            if error.status_code is not None
            else ""
        )
        raise RuntimeError(
            f"Ollama returned an error{status}: {error.error}"
        ) from error
    except Exception as error:
        raise RuntimeError(
            "Could not complete the local Ollama request. "
            "Confirm that Ollama is running and "
            "qwen3:4b-instruct is installed. "
            f"Original error: {error}"
        ) from error

    content = str(response.message.content or "").strip()

    if not content:
        raise ValueError("Ollama returned no final report")

    print("[AI] Final report received.", flush=True)

    return content