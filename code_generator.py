import json
import os
import re
from typing import Any

from pydantic import BaseModel

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


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


# LangChain prompt templates treat `{...}` as variables, so we escape braces to
# avoid accidental interpolation if the prompt ever contains JSON examples.
SYSTEM_PROMPT = RAW_SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")


class GeneratedCodeFile(BaseModel):
	file_name: str
	code: str


def build_parser() -> PydanticOutputParser:
	return PydanticOutputParser(pydantic_object=GeneratedCodeFile)


def _extract_first_json_object(text: str) -> str:
	"""Best-effort: extract the first top-level JSON object from text."""
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
	"""Parse a generated output payload into a `GeneratedCodeFile`.

	Accepts either:
	- a dict like {"file_name": "main.py", "code": "..."}
	- a string containing that JSON (even if it has extra text around it)
	"""
	if isinstance(payload, dict):
		data = payload
	else:
		json_text = _extract_first_json_object(payload)
		data = json.loads(json_text)

	generated = GeneratedCodeFile.model_validate(data)
	generated.code = _strip_markdown_fences(generated.code)
	return generated


def write_generated_code_to_file(
	generated: GeneratedCodeFile,
	*,
	out_dir: str = ".",
	overwrite: bool = True,
) -> str:
	"""Write `generated.code` into `out_dir/generated.file_name`.

	Returns the absolute path of the written file.
	"""
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
	"""Convenience: parse payload then write it to disk."""
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
				"- The `code` field must contain plain code text.\n",
			),
		]
	)


def build_model(
	*,
	model_name: str = "qwen3:8b",
	temperature: float = 0.0,
	output_format: str = "json",
) -> ChatOllama:
	return ChatOllama(
		model=model_name,
		validate_model_on_init=True,
		format=output_format,
		temperature=temperature,
	)


def build_chain(*, system_prompt: str = SYSTEM_PROMPT) -> tuple[object, PydanticOutputParser]:
	parser = build_parser()
	prompt = build_prompt_template(system_prompt)
	model = build_model()
	chain = prompt | model | parser
	return chain, parser


def generate_code(problem_spec: dict) -> GeneratedCodeFile:
	chain, parser = build_chain()
	result: GeneratedCodeFile = chain.invoke(
		{
			"problem_spec_json": json.dumps(problem_spec, indent=2, ensure_ascii=False),
			"format_instructions": parser.get_format_instructions(),
		}
	)

	# Extra safety: ensure no accidental Markdown fences leak through.
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
	
	# Write the generated code to disk
	output_path = write_generated_code_to_file(result, out_dir="real", overwrite=True)
	print(f"\n✓ Wrote generated code to: {output_path}")


if __name__ == "__main__":
	main()

