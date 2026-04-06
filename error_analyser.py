from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json


SYSTEM_PROMPT = """You are an Error Analyzer.

Input:
- Failed test cases
- Code

Task:
- Identify the root cause of each failure.
- Classify errors (logic error, boundary condition, incorrect assumption, etc.).

Rules:
- Be precise.
- Do NOT propose fixes yet.

Output format:
{
  "error_summary": "",
  "error_categories": [],
  "root_causes": []
}"""


def analyze_errors(
    code: str, failed_test_cases: list[dict], model: str = "gemma4:e4b"
) -> dict:
    """
    Analyze failed test cases and identify root causes of errors.

    Args:
        code: The source code that was tested
        failed_test_cases: List of failed test cases with their details
        model: Ollama model to use for analysis

    Returns:
        Dictionary containing error_summary, error_categories, and root_causes
    """
    llm = ChatOllama(model=model)

    user_message = f"""Code:
```
{code}
```

Failed Test Cases:
{json.dumps(failed_test_cases, indent=2)}

Analyze the errors and provide the output in the specified JSON format."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    content_raw = response.content if hasattr(response, "content") else response
    if isinstance(content_raw, list):
        response_text = "\n".join(
            item if isinstance(item, str) else str(item) for item in content_raw
        )
    else:
        response_text = str(content_raw)

    # Try to parse JSON from the response
    try:
        # Find JSON in the response (it might be wrapped in markdown code blocks)
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            json_str = response_text.strip()

        result = json.loads(json_str)
    except json.JSONDecodeError:
        # Return raw response if JSON parsing fails
        result = {
            "error_summary": response_text,
            "error_categories": [],
            "root_causes": [],
            "parse_error": "Could not parse JSON from model response",
        }

    return result


if __name__ == "__main__":
    # Example usage
    sample_code = """
def add(a, b):
    return a - b  # Bug: should be a + b
"""

    sample_failed_tests = [
        {
            "test_name": "test_add_positive",
            "input": {"a": 2, "b": 3},
            "expected": 5,
            "actual": -1,
            "error": "AssertionError: Expected 5 but got -1",
        }
    ]

    result = analyze_errors(sample_code, sample_failed_tests)
    print(json.dumps(result, indent=2))
