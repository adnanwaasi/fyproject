import json
import os
import re
from typing import Any

from pydantic import BaseModel

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from groq_llm import build_groq_model, invoke_with_retry, DEFAULT_MODEL


RAW_SYSTEM_PROMPT = """You are a Code Generator.

Input:
- A structured problem specification in JSON.
- You must generate code that satisfies the specification exactly.

Rules:
- Use clear, readable, production-style code.
- Do NOT include explanations.
- Do NOT include test cases.
- Follow all constraints strictly.
- If assumptions exist, reflect them in code comments.

Target language: Python

Output:
- Output must be structured JSON with fixed keys.
- The code must be plain text in the JSON (no Markdown fences like ```python).
- Include the target filename.

Output format:
{
	"file_name": "main.py",
	"code": "..."
}
"""


SYSTEM_PROMPT = RAW_SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")


class GeneratedCodeFile(BaseModel):
    file_name: str
    code: str


def build_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=GeneratedCodeFile)


def _extract_first_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return match.group(0)


def _strip_markdown_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", code)
        code = re.sub(r"\n```\s*$", "", code)
    return code


def parse_generated_code(payload: str | dict[str, Any]) -> GeneratedCodeFile:
    if isinstance(payload, dict):
        data = payload.copy()
    else:
        json_text = _extract_first_json_object(payload)
        data = json.loads(json_text)

    if "file_name" not in data or not data.get("file_name"):
        data["file_name"] = "main.py"

    generated = GeneratedCodeFile.model_validate(data)
    generated.code = _strip_markdown_fences(generated.code)
    return generated


def write_generated_code_to_file(
    generated: GeneratedCodeFile,
    *,
    out_dir: str = ".",
    overwrite: bool = True,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, generated.file_name))

    if not overwrite and os.path.exists(out_path):
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    content = generated.code
    if content and not content.endswith("\n"):
        content += "\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    return out_path


def parse_and_write_generated_code(
    payload: str | dict[str, Any],
    *,
    out_dir: str = ".",
    overwrite: bool = True,
) -> str:
    generated = parse_generated_code(payload)
    return write_generated_code_to_file(generated, out_dir=out_dir, overwrite=overwrite)


def build_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Problem specification (JSON):\n{problem_spec_json}\n\n{format_instructions}\n\n"
                "IMPORTANT:\n"
                "- Output MUST be a single JSON object and nothing else.\n"
                "- Do NOT use Markdown. Do NOT include ``` fences.\n"
                "- The `code` field must contain plain code text with proper JSON escaping:\n"
                '  * Use \\\\n for newlines, \\\\t for tabs, \\\\" for double quotes, \\\\\\\\ for backslashes\n'
                "- Do NOT include unescaped newlines or quotes within the code string.\n"
                "- The entire response must be valid JSON that can be parsed by Python's json.loads().\n",
            ),
        ]
    )


def generate_code(
    problem_spec: dict, model_name: str = DEFAULT_MODEL
) -> GeneratedCodeFile:
    parser = build_parser()
    prompt = build_prompt_template(SYSTEM_PROMPT)
    model = build_groq_model(model_name=model_name, json_mode=True)

    messages = prompt.format_messages(
        problem_spec_json=json.dumps(problem_spec, indent=2, ensure_ascii=False),
        format_instructions=parser.get_format_instructions(),
    )
    raw_result = invoke_with_retry(model, messages)
    content: str = (
        raw_result.content if hasattr(raw_result, "content") else str(raw_result)
    )

    result = parse_generated_code(content)
    result.code = _strip_markdown_fences(result.code)
    return result


def main() -> None:
    example_problem_spec = {
        "problem_summary": "Build a CLI tool that reads a CSV file and outputs duplicates.",
        "inputs": ["Path to a CSV file"],
        "outputs": ["Printed report or JSON list of duplicate rows"],
        "constraints": [
            "Must be a CLI program",
            "No test cases in output",
            "Handle missing/empty CSV gracefully",
        ],
        "edge_cases": [
            "CSV file not found",
            "CSV with headers missing",
            "No duplicates found",
        ],
        "assumptions": [
            "Duplicates are determined by exact row match",
            "CSV is UTF-8 encoded",
        ],
    }

    result = generate_code(example_problem_spec)
    print(result.model_dump_json(indent=2))

    output_path = write_generated_code_to_file(result, out_dir="real", overwrite=True)
    print(f"\n✓ Wrote generated code to: {output_path}")


if __name__ == "__main__":
    main()
