import json
from typing import List

from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


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
"""


SYSTEM_PROMPT = RAW_SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")


class TestCase(BaseModel):
    test_id: str
    description: str
    test_type: str  # normal, edge_case, error
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


def build_model(*, model_name: str = "qwen3:8b", temperature: float = 0.3) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        validate_model_on_init=True,
        format="json",
        temperature=temperature,
    )


def build_chain(*, system_prompt: str = SYSTEM_PROMPT) -> tuple[object, PydanticOutputParser]:
    parser = build_parser()
    prompt = build_prompt_template(system_prompt)
    model = build_model()
    chain = prompt | model | parser
    return chain, parser


def generate_test_cases(problem_spec: dict | BaseModel) -> TestCaseCollection:
    """Generate test cases from a problem specification.
    
    Args:
        problem_spec: Either a dict or ProblemSpecification object
    
    Returns:
        TestCaseCollection with generated test cases
    """
    chain, parser = build_chain()
    
    # Convert to dict if it's a Pydantic model
    if hasattr(problem_spec, "model_dump"):
        spec_dict = problem_spec.model_dump()
    else:
        spec_dict = problem_spec
    
    result: TestCaseCollection = chain.invoke(
        {
            "problem_spec": json.dumps(spec_dict, indent=2, ensure_ascii=False),
            "format_instructions": parser.get_format_instructions(),
        }
    )
    
    return result


def main() -> None:
    # Example problem specification
    example_spec = {
        "problem_summary": "Build a CLI tool to read a CSV file and compute summary statistics (mean, median, mode) for specified columns, handling missing values and allowing column selection via command-line arguments.",
        "inputs": [
            "CSV file path",
            "List of column names to analyze (specified via command-line arguments)"
        ],
        "outputs": [
            "JSON-formatted summary statistics for each specified column (mean, median, mode)",
            "Error messages for invalid inputs or processing issues"
        ],
        "constraints": [
            "Gracefully handle missing values in the dataset",
            "Support statistical calculations only for numeric columns",
            "Allow dynamic column selection via command-line arguments"
        ],
        "edge_cases": [
            "Missing or invalid CSV file path",
            "Specified columns do not exist in the CSV",
            "Non-numeric data in columns requiring numerical operations",
            "All values in a column are missing"
        ],
        "assumptions": [
            "The CSV file uses standard formatting with headers",
            "Command-line arguments for columns are valid strings",
            "The tool is used in an environment supporting CLI arguments",
            "The user intends to analyze numeric columns for statistical calculations"
        ]
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
    
    # Also output as JSON
    print("\n\nJSON Output:")
    print(test_cases.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
