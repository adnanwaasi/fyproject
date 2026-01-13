import json

from langchain_core.output_parsers import StrOutputParser
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
- Single complete code file.
"""


# LangChain prompt templates treat `{...}` as variables, so we escape braces to
# avoid accidental interpolation if the prompt ever contains JSON examples.
SYSTEM_PROMPT = RAW_SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")


prompt = ChatPromptTemplate.from_messages(
	[
		("system", SYSTEM_PROMPT),
		(
			"human",
			"Problem specification (JSON):\n{problem_spec_json}\n\nReturn only the code file contents.",
		),
	]
)


model = ChatOllama(
	model="qwen3:14b",
	validate_model_on_init=True,
	temperature=0.0,
)


chain = prompt | model | StrOutputParser()


if __name__ == "__main__":
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

	result = chain.invoke(
		{
			"problem_spec_json": json.dumps(example_problem_spec, indent=2, ensure_ascii=False),
		}
	)
	print(result)

