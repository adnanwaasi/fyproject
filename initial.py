from typing import Any, List

from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


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


# LangChain prompt templates treat `{...}` as variables, so we escape braces to
# ensure the JSON example is sent literally.
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


def build_model(
    *, model_name: str = "gemma4:e4b", temperature: float = 0.0
) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        validate_model_on_init=True,
        temperature=temperature,
    )


def build_chain(
    *, system_prompt: str = SYSTEM_PROMPT
) -> tuple[Any, PydanticOutputParser]:
    parser = build_parser()
    prompt = build_prompt_template(system_prompt)
    model = build_model()
    chain = prompt | model | parser
    return chain, parser


def process_user_input(user_input: str) -> ProblemSpecification:
    chain, parser = build_chain()
    return chain.invoke(
        {
            "user_input": user_input,
            "format_instructions": parser.get_format_instructions(),
        }
    )


def main() -> None:
    result = process_user_input(
        "Build a CLI that reads a CSV of contacts and outputs duplicates."
    )
    print(result)


if __name__ == "__main__":
    main()
