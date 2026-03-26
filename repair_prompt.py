from langchain_core.messages import SystemMessage, HumanMessage
import json
import re

from groq_llm import build_groq_model, invoke_with_retry, DEFAULT_MODEL


SYSTEM_PROMPT = """You are a Code Repair Agent.

Input:
- Original buggy code
- Error analysis containing:
  - error_summary: Brief description of the bug
  - error_categories: Types of errors (e.g., logic error, boundary condition)
  - root_causes: Specific reasons for the failures

Task:
- Fix the code based on the error analysis.
- Address ALL identified root causes.
- Preserve the original code structure and style as much as possible.

Rules:
- Output ONLY the corrected code.
- Do NOT include explanations or comments about what you changed.
- Do NOT include test cases.
- Keep the same function signatures and names.
- Ensure the fix addresses each root cause mentioned.

Output format:
{
  "repaired_code": "..."
}"""


def _extract_json_from_response(response_text: str) -> str:
    if "```json" in response_text:
        json_start = response_text.find("```json") + 7
        json_end = response_text.find("```", json_start)
        return response_text[json_start:json_end].strip()
    elif "```" in response_text:
        json_start = response_text.find("```") + 3
        json_end = response_text.find("```", json_start)
        return response_text[json_start:json_end].strip()
    else:
        match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        if match:
            return match.group(0)
        return response_text.strip()


def _strip_markdown_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", code)
        code = re.sub(r"\n?```\s*$", "", code)
    return code


def repair_code(code: str, error_analysis: dict, model: str = DEFAULT_MODEL) -> dict:
    llm = build_groq_model(model_name=model)

    user_message = f"""Original Code:
```
{code}
```

Error Analysis:
{json.dumps(error_analysis, indent=2)}

Please fix the code based on the error analysis and provide the repaired code in the specified JSON format."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = invoke_with_retry(llm, messages)
    response_text: str = (
        response.content if hasattr(response, "content") else str(response)
    )

    try:
        json_str = _extract_json_from_response(response_text)
        result = json.loads(json_str)

        if "repaired_code" in result:
            result["repaired_code"] = _strip_markdown_fences(result["repaired_code"])

    except json.JSONDecodeError:
        code_match = re.search(r"```python\n(.*?)```", response_text, flags=re.DOTALL)
        if code_match:
            result = {"repaired_code": code_match.group(1).strip()}
        else:
            result = {
                "repaired_code": response_text,
                "parse_error": "Could not parse JSON from model response",
            }

    return result


if __name__ == "__main__":
    sample_code = """
def add(a, b):
    return a - b  # Bug: should be a + b
"""

    sample_error_analysis = {
        "error_summary": "The function incorrectly subtracts parameters instead of adding them, leading to incorrect results.",
        "error_categories": ["logic error"],
        "root_causes": ["Incorrect operator usage (subtraction instead of addition)"],
    }

    print("Original Code:")
    print(sample_code)
    print("\nError Analysis:")
    print(json.dumps(sample_error_analysis, indent=2))
    print("\nRepairing code...")

    result = repair_code(sample_code, sample_error_analysis)

    print("\nRepair Result:")
    print(json.dumps(result, indent=2))
