from typing import List

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


parser = PydanticOutputParser(pydantic_object=ProblemSpecification)


# 3. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    (
        "human",
        "User input:\n{user_input}\n\n{format_instructions}",
    ),
])


# 4. Model
model = ChatOllama(
    model="qwen3:14b",
    validate_model_on_init=True,
    temperature=0.0,
)


# 5. Chain
chain = prompt | model | parser


if __name__ == "__main__":
    result = chain.invoke(
        {
            "user_input": "Build a CLI tool that reads a CSV file and outputs summary statistics like mean, median, and mode for specified columns. The tool should handle missing values gracefully and allow users to specify which columns to analyze via command-line arguments.",
            "format_instructions": parser.get_format_instructions(),
        }
    )
    print(result)