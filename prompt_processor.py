from typing import List

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from groq_llm import build_groq_model, invoke_with_retry, DEFAULT_MODEL


RAW_SYSTEM_PROMPT = """You are a Prompt Processor for a code synthesis system.

Your task:
- Extract an explicit problem specification from the user's input.
- Identify inputs, outputs, constraints, edge cases, and assumptions.
- Rewrite the problem as a formal specification.

Rules:
- Do NOT generate code.
- Do NOT infer missing requirements silently.
- If information is missing, state it explicitly as an assumption.
- Output must be structured JSON with fixed keys.

Output format:
{
    "problem_summary": "",
    "inputs": [],
    "outputs": [],
    "constraints": [],
    "edge_cases": [],
    "assumptions": []
}
"""


SYSTEM_PROMPT = RAW_SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")


class ProblemSpecification(BaseModel):
    problem_summary: str
    inputs: List[str]
    outputs: List[str]
    constraints: List[str]
    edge_cases: List[str]
    assumptions: List[str]


def build_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=ProblemSpecification)


def build_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "User input:\n{user_input}\n\n{format_instructions}",
            ),
        ]
    )


def process_user_input(
    user_input: str, model_name: str = DEFAULT_MODEL
) -> ProblemSpecification:
    parser = build_parser()
    prompt = build_prompt_template(SYSTEM_PROMPT)
    model = build_groq_model(model_name=model_name, json_mode=True)

    messages = prompt.format_messages(
        user_input=user_input,
        format_instructions=parser.get_format_instructions(),
    )
    response = invoke_with_retry(model, messages)
    content = response.content if hasattr(response, "content") else str(response)
    return parser.parse(content)


def main() -> None:
    result = process_user_input(
        "Build a CLI tool that reads a CSV file and outputs summary statistics like mean, median, and mode for specified columns. The tool should handle missing values gracefully and allow users to specify which columns to analyze via command-line arguments."
    )
    print(result)


if __name__ == "__main__":
    main()
