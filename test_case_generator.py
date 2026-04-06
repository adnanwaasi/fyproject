import json
import ollama
from typing import List, Optional
from collections import OrderedDict

from pydantic import BaseModel


class TestCase(BaseModel):
    test_id: str
    description: str
    test_type: str
    input_data: dict
    expected_output: str
    validation_criteria: str


class TestCaseCollection(BaseModel):
    test_cases: List[TestCase]


SYSTEM_PROMPT = """You are a Test Case Generator for a code synthesis system.

Generate test cases that validate the ALREADY GENERATED CODE.
Cover: normal usage, edge cases, error handling, and boundary conditions.

CRITICAL: The input_data format depends on the code structure:

1. If code has a CLASS with methods (like LRUCache with get/put):
   input_data must be: {"capacity": N, "operations": [{"op": "method_name", "arg1": val1, ...}]}
   Example for LRUCache: {"capacity": 2, "operations": [{"op": "put", "key": 1, "value": 10}, {"op": "get", "key": 1}]}
   expected_output should be a list of return values from "get" operations: "[10]"

2. If code has standalone FUNCTIONS:
   input_data should be direct arguments: {"n": 5, "x": 10}
   expected_output should be the function's return value as string: "120"

Output as JSON with test_cases array.
Each test case needs: test_id, description, test_type (normal/edge_case/error), input_data, expected_output, validation_criteria.
"""


def normalize_test_case(tc_data: dict) -> dict:
    normalized = tc_data.copy()
    if "id" in normalized and "test_id" not in normalized:
        normalized["test_id"] = normalized.pop("id")
    if "test_id" not in normalized:
        normalized["test_id"] = f"TC{id(normalized) % 1000:03d}"

    # Normalize test_type
    tt = normalized.get("test_type", "normal").lower()
    if tt in ["boundary", "edge"]:
        normalized["test_type"] = "edge_case"
    elif tt not in ["normal", "edge_case", "error"]:
        normalized["test_type"] = "normal"

    if "description" not in normalized:
        normalized["description"] = "Test case"
    if "input_data" not in normalized:
        normalized["input_data"] = {}
    if "expected_output" not in normalized:
        normalized["expected_output"] = ""
    else:
        normalized["expected_output"] = str(normalized["expected_output"])
    if "validation_criteria" not in normalized:
        normalized["validation_criteria"] = (
            "Verify expected output matches actual output"
        )

    input_data = normalized.get("input_data", {})
    if (
        isinstance(input_data, dict)
        and "capacity" in input_data
        and "operations" in input_data
    ):
        expected_from_ops = derive_lru_expected_output(input_data)
        if expected_from_ops is not None:
            normalized["expected_output"] = str(expected_from_ops)

    return normalized


def derive_lru_expected_output(input_data: dict) -> Optional[list]:
    capacity = input_data.get("capacity")
    operations = input_data.get("operations")
    if not isinstance(capacity, int) or not isinstance(operations, list):
        return None
    if capacity <= 0:
        return None

    cache: OrderedDict = OrderedDict()
    out: list = []

    for op in operations:
        if not isinstance(op, dict):
            continue
        op_type = str(op.get("op", "")).lower()
        key = op.get("key")
        if op_type == "put":
            value = op.get("value")
            if key in cache:
                cache.move_to_end(key)
                cache[key] = value
            else:
                if len(cache) >= capacity:
                    cache.popitem(last=False)
                cache[key] = value
        elif op_type == "get":
            if key in cache:
                cache.move_to_end(key)
                out.append(cache[key])
            else:
                out.append(-1)

    return out


def parse_test_cases(raw_content: str | dict) -> TestCaseCollection:
    import re

    if isinstance(raw_content, str):
        raw_content = raw_content.strip()
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_content, flags=re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = {"test_cases": []}
            else:
                data = {"test_cases": []}
    else:
        data = raw_content

    test_cases_raw = data.get("test_cases", [])
    if not test_cases_raw:
        test_cases_raw = [
            {
                "test_id": "TC001",
                "description": "Default test",
                "test_type": "normal",
                "input_data": {},
                "expected_output": "",
                "validation_criteria": "",
            }
        ]

    normalized_cases = [normalize_test_case(tc) for tc in test_cases_raw]
    test_cases = [TestCase.model_validate(tc) for tc in normalized_cases]
    return TestCaseCollection(test_cases=test_cases)


def generate_test_cases(
    problem_spec: dict | BaseModel,
    generated_code: Optional[str] = None,
    model_name: str = "gemma4:e4b",
) -> TestCaseCollection:
    if isinstance(problem_spec, BaseModel):
        spec_dict = problem_spec.model_dump()
    else:
        spec_dict = problem_spec

    spec_json = json.dumps(spec_dict, indent=2, ensure_ascii=False)

    if generated_code:
        prompt = (
            "Analyze this Python code and generate test cases that EXACTLY match its API.\n\n"
            "CODE TO TEST:\n" + generated_code + "\n\n"
            "PROBLEM SPEC:\n" + spec_json + "\n\n"
            "CRITICAL RULES:\n"
            "1. Look at the class/function names and signatures in the code above.\n"
            "2. If the code defines a CLASS with methods (e.g., LRUCache with get/put):\n"
            '   - input_data MUST be: {"capacity": N, "operations": [{"op": "method_name", "arg1": val1, ...}]}\n'
            '   - expected_output MUST be a JSON list of return values from query methods: "[1, -1]"\n'
            '   - Example: {"capacity": 2, "operations": [{"op": "put", "key": 1, "value": 10}, {"op": "get", "key": 1}]}\n'
            "3. If the code defines standalone FUNCTIONS:\n"
            '   - input_data should be direct arguments matching the function signature: {"n": 5}\n'
            '   - expected_output should be the return value as a JSON string: "120"\n'
            "4. Generate 5-8 test cases covering: normal usage, edge cases, boundary conditions.\n"
            "5. Output ONLY valid JSON with this exact structure:\n"
            '{"test_cases": [{"test_id": "TC001", "description": "...", "test_type": "normal", "input_data": {}, "expected_output": "...", "validation_criteria": "..."}]}'
        )
    else:
        prompt = (
            "Generate 5 test cases for: "
            + spec_json
            + ". Output ONLY JSON with test_cases array."
        )

    try:
        resp = ollama.generate(
            model=model_name, prompt=prompt, format="json", options={"temperature": 0.3}
        )
        content = resp.get("response", "")
    except Exception as e:
        print(f"Ollama error: {e}")
        content = '{"test_cases": []}'

    return parse_test_cases(content)


def main():
    code = """
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
    def get(self, key: int) -> int:
        if key not in self.cache: return -1
        val = self.cache.pop(key)
        self.cache[key] = val
        return val
    def put(self, key: int, value: int) -> None:
        if key in self.cache: self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = value
"""
    spec = {
        "problem_summary": "LRU Cache",
        "inputs": [],
        "outputs": [],
        "constraints": [],
        "edge_cases": [],
        "assumptions": [],
    }

    print("Generating test cases...")
    tc = generate_test_cases(spec, generated_code=code)
    print(f"Generated {len(tc.test_cases)} test cases")
    for t in tc.test_cases:
        print(f"  {t.test_id}: {t.test_type} - {t.description[:40]}...")


if __name__ == "__main__":
    main()
