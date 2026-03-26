import json
from typing import List

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from groq_llm import build_groq_model, invoke_with_retry, DEFAULT_MODEL


RAW_SYSTEM_PROMPT = """You are a Test Case Generator for a code synthesis system.

Your task:
- Generate comprehensive test cases based on a problem specification.
- Cover normal cases, edge cases, and error scenarios.
- Each test case must be executable and verifiable.

Rules:
- Generate realistic test inputs and expected outputs.
- Cover all edge cases mentioned in the specification.
- Include both positive and negative test cases.
- Test cases should be clear and unambiguous.
- Output must be structured JSON with EXACTLY these field names.

CRITICAL: Each test case MUST have these exact fields:
- test_id (string): unique identifier like "TC001"
- description (string): what this test validates
- test_type (string): must be "normal", "edge_case", or "error"
- input_data (object): test inputs as JSON object
- expected_output (string): expected behavior or result
- validation_criteria (string): how to verify this test passes

Output format:
{{
    "test_cases": [
        {{
            "test_id": "TC001",
            "description": "Brief description of what this test validates",
            "test_type": "normal",
            "input_data": {{}},
            "expected_output": "description of expected behavior or output",
            "validation_criteria": "how to verify this test passes"
        }}
    ]
}}

IMPORTANT:
- Output MUST be a single JSON object and nothing else.
- Do NOT use Markdown. Do NOT include ``` fences.
- The entire response must be valid JSON that can be parsed by Python's json.loads().
- Escape special characters in strings appropriately for JSON.
"""


SYSTEM_PROMPT = RAW_SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")


class TestCase(BaseModel):
    test_id: str
    description: str
    test_type: str
    input_data: dict
    expected_output: str
    validation_criteria: str


class TestCaseCollection(BaseModel):
    test_cases: List[TestCase]


def build_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=TestCaseCollection)


def build_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Problem Specification:\n{problem_spec}\n\n{format_instructions}\n\n"
                "CRITICAL REQUIREMENTS:\n"
                "- Generate at least 5-10 diverse test cases.\n"
                "- Cover all edge cases mentioned in the specification.\n"
                "- Each test case MUST have ALL required fields: test_id, description, test_type, input_data, expected_output, validation_criteria\n"
                "- test_type must be one of: normal, edge_case, error\n"
                "- Output MUST be valid JSON only with proper field names.\n",
            ),
        ]
    )


def normalize_test_case(tc_data: dict) -> dict:
    normalized = tc_data.copy()

    if "id" in normalized and "test_id" not in normalized:
        normalized["test_id"] = normalized.pop("id")

    if "test_id" not in normalized:
        normalized["test_id"] = f"TC{id(normalized) % 1000:03d}"

    if "test_type" not in normalized:
        desc = normalized.get("description", "").lower()
        expected = normalized.get("expected_output", "").lower()

        if any(
            word in desc
            for word in ["error", "invalid", "fail", "exception", "negative"]
        ):
            normalized["test_type"] = "error"
        elif any(
            word in desc
            for word in ["edge", "boundary", "empty", "zero", "max", "min", "limit"]
        ):
            normalized["test_type"] = "edge_case"
        elif any(
            word in expected for word in ["error", "-1", "none", "null", "invalid"]
        ):
            normalized["test_type"] = "edge_case"
        else:
            normalized["test_type"] = "normal"

    if "description" not in normalized:
        normalized["description"] = "Test case"
    if "input_data" not in normalized:
        normalized["input_data"] = {}
    if "expected_output" not in normalized:
        normalized["expected_output"] = ""
    if "validation_criteria" not in normalized:
        normalized["validation_criteria"] = (
            "Verify expected output matches actual output"
        )

    return normalized


def parse_test_cases(raw_content: str | dict) -> TestCaseCollection:
    import re

    if isinstance(raw_content, str):
        raw_content = raw_content.strip()
        match = re.search(r"\{.*\}", raw_content, flags=re.DOTALL)
        if match:
            raw_content = match.group(0)
        data = json.loads(raw_content)
    else:
        data = raw_content

    if "test_cases" in data:
        test_cases_raw = data["test_cases"]
    else:
        test_cases_raw = data if isinstance(data, list) else [data]

    normalized_cases = [normalize_test_case(tc) for tc in test_cases_raw]
    test_cases = [TestCase.model_validate(tc) for tc in normalized_cases]
    return TestCaseCollection(test_cases=test_cases)


def generate_test_cases(
    problem_spec: dict | BaseModel, model_name: str = DEFAULT_MODEL
) -> TestCaseCollection:
    parser = build_parser()
    prompt = build_prompt_template(SYSTEM_PROMPT)
    model = build_groq_model(model_name=model_name, json_mode=True)

    spec_dict = (
        problem_spec.model_dump()
        if hasattr(problem_spec, "model_dump")
        else problem_spec
    )

    messages = prompt.format_messages(
        problem_spec=json.dumps(spec_dict, indent=2, ensure_ascii=False),
        format_instructions=parser.get_format_instructions(),
    )
    raw_result = invoke_with_retry(model, messages)
    content: str = (
        raw_result.content if hasattr(raw_result, "content") else str(raw_result)
    )
    return parse_test_cases(content)


def main() -> None:
    example_spec = {
        "problem_summary": "Build a CLI tool to read a CSV file and compute summary statistics (mean, median, mode) for specified columns, handling missing values and allowing column selection via command-line arguments.",
        "inputs": [
            "CSV file path",
            "List of column names to analyze (specified via command-line arguments)",
        ],
        "outputs": [
            "JSON-formatted summary statistics for each specified column (mean, median, mode)",
            "Error messages for invalid inputs or processing issues",
        ],
        "constraints": [
            "Gracefully handle missing values in the dataset",
            "Support statistical calculations only for numeric columns",
            "Allow dynamic column selection via command-line arguments",
        ],
        "edge_cases": [
            "Missing or invalid CSV file path",
            "Specified columns do not exist in the CSV",
            "Non-numeric data in columns requiring numerical operations",
            "All values in a column are missing",
        ],
        "assumptions": [
            "The CSV file uses standard formatting with headers",
            "Command-line arguments for columns are valid strings",
            "The tool is used in an environment supporting CLI arguments",
            "The user intends to analyze numeric columns for statistical calculations",
        ],
    }

    test_cases = generate_test_cases(example_spec)

    print("Generated Test Cases:")
    print("=" * 80)
    for tc in test_cases.test_cases:
        print(f"\n{tc.test_id}: {tc.description}")
        print(f"Type: {tc.test_type}")
        print(f"Input: {json.dumps(tc.input_data, indent=2)}")
        print(f"Expected: {tc.expected_output}")
        print(f"Validation: {tc.validation_criteria}")
        print("-" * 80)

    print("\n\nJSON Output:")
    print(test_cases.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
