import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel


class TestCase(BaseModel):
    test_id: str
    description: str
    test_type: str
    input_data: dict
    expected_output: str
    validation_criteria: str


class TestResult(BaseModel):
    test_id: str
    passed: bool
    actual_output: str
    expected_output: str
    error_message: Optional[str] = None
    execution_time: float = 0.0


def find_main_file(search_dir: str = "real") -> Optional[Path]:
    """Find the main Python file in the specified directory."""
    search_path = Path(search_dir)
    
    if not search_path.exists():
        return None
    
    # Look for common entry point names
    priority_files = ["main.py", "app.py", "cli.py", "solution.py"]
    
    for filename in priority_files:
        file_path = search_path / filename
        if file_path.exists():
            return file_path
    
    # If no priority file found, return first .py file
    py_files = list(search_path.glob("*.py"))
    if py_files:
        return py_files[0]
    
    return None


def prepare_test_input(test_case: TestCase) -> tuple[list[str], Optional[Path]]:
    """Prepare command-line arguments and any required input files for test execution.
    
    Returns:
        tuple: (command_args, temp_file_path)
    """
    args = []
    temp_file = None
    
    # Extract common input patterns
    input_data = test_case.input_data
    
    # Handle file_path argument
    if "file_path" in input_data:
        file_path = input_data["file_path"]
        
        # If test expects file to exist and it's not 'nonexistent', create temp file
        if test_case.test_type == "normal" and "nonexistent" not in file_path.lower():
            # Create a temporary CSV file with sample data
            temp_file = create_sample_csv(file_path, input_data.get("columns", []))
            args.append(temp_file)
        else:
            args.append(file_path)
    
    # Handle columns argument
    if "columns" in input_data:
        columns = input_data["columns"]
        if isinstance(columns, list):
            args.extend(columns)
        else:
            args.append(str(columns))
    
    # Handle any other string/simple arguments
    for key, value in input_data.items():
        if key not in ["file_path", "columns"]:
            if isinstance(value, list):
                args.extend([str(v) for v in value])
            else:
                args.append(str(value))
    
    return args, temp_file


def create_sample_csv(filename: str, columns: list[str]) -> str:
    """Create a sample CSV file for testing."""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    
    # Create sample data based on column names
    with open(temp_path, "w") as f:
        # Write header
        f.write(",".join(columns) + "\n")
        
        # Write sample data rows
        for i in range(10):
            row_data = []
            for col in columns:
                # Generate realistic sample data
                if "age" in col.lower():
                    row_data.append(str(25 + i * 2))
                elif "income" in col.lower() or "salary" in col.lower():
                    row_data.append(str(45000 + i * 2000))
                elif "score" in col.lower():
                    row_data.append(str(70 + i * 3))
                elif "price" in col.lower():
                    row_data.append(str(10.5 + i * 1.5))
                else:
                    row_data.append(str(i * 10))
            f.write(",".join(row_data) + "\n")
    
    return temp_path


def execute_test(main_file: Path, test_case: TestCase, timeout: int = 10) -> TestResult:
    """Execute a single test case and return the result.
    
    This function supports two modes:
    1. Function-based: Import the module and call functions directly
    2. CLI-based: Run as subprocess with command-line arguments
    """
    import time
    import importlib.util
    
    start_time = time.time()
    
    try:
        # First, try to import and execute as a module with functions
        result = execute_as_function(main_file, test_case, timeout)
        if result is not None:
            return result
        
        # Fall back to CLI execution
        return execute_as_cli(main_file, test_case, timeout, start_time)
        
    except Exception as e:
        return TestResult(
            test_id=test_case.test_id,
            passed=False,
            actual_output="",
            expected_output=test_case.expected_output,
            error_message=f"Execution error: {str(e)}",
            execution_time=time.time() - start_time
        )


def execute_as_function(main_file: Path, test_case: TestCase, timeout: int = 10) -> Optional[TestResult]:
    """Try to execute test by importing module and calling function directly."""
    import time
    import importlib.util
    
    start_time = time.time()
    
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location("test_module", main_file)
        if spec is None or spec.loader is None:
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find callable functions in the module
        functions = {name: obj for name, obj in vars(module).items() 
                    if callable(obj) and not name.startswith('_')}
        
        if not functions:
            return None
        
        # Get the main function (first one or one matching common patterns)
        func_name = None
        input_data = test_case.input_data
        
        # Check if input_data specifies a function name
        if "function" in input_data:
            func_name = input_data["function"]
        else:
            # Try to find the main function
            priority_names = ["fibonacci", "fib", "main", "solve", "solution", "run", "execute"]
            for name in priority_names:
                if name in functions:
                    func_name = name
                    break
            
            # Also check for partial matches (e.g., fibonacci_series)
            if func_name is None:
                for name in functions:
                    if any(p in name.lower() for p in ["fibonacci", "fib"]):
                        func_name = name
                        break
            
            # If no priority name, use the first function
            if func_name is None:
                func_name = list(functions.keys())[0]
        
        if func_name not in functions:
            return None
        
        func = functions[func_name]
        
        # Prepare arguments from input_data
        args = []
        kwargs = {}
        
        # Common parameter names for sequence/math functions
        positional_params = ["n", "num", "number", "count", "terms", "limit", "input", "value", "x"]
        
        # Check for positional parameters first
        found_positional = False
        for param in positional_params:
            if param in input_data:
                val = input_data[param]
                if isinstance(val, list):
                    args.extend(val)
                else:
                    args.append(val)
                found_positional = True
                break
        
        if not found_positional:
            # Pass all input_data as kwargs (excluding metadata fields)
            for key, value in input_data.items():
                if key not in ["function", "file_path"]:
                    kwargs[key] = value
        
        # Execute the function
        try:
            if kwargs:
                actual_result = func(**kwargs)
            elif args:
                actual_result = func(*args)
            else:
                actual_result = func()
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_id=test_case.test_id,
                passed=test_case.test_type == "error",
                actual_output=str(e),
                expected_output=test_case.expected_output,
                error_message=str(e) if test_case.test_type != "error" else None,
                execution_time=execution_time
            )
        
        execution_time = time.time() - start_time
        
        # Convert result to string for comparison
        if isinstance(actual_result, (list, dict)):
            actual_output = json.dumps(actual_result)
        else:
            actual_output = str(actual_result)
        
        # Verify output
        passed = verify_output(
            actual_output,
            test_case.expected_output,
            test_case.test_type
        )
        
        return TestResult(
            test_id=test_case.test_id,
            passed=passed,
            actual_output=actual_output,
            expected_output=test_case.expected_output,
            execution_time=execution_time
        )
        
    except Exception as e:
        # If function execution fails, return None to try CLI mode
        return None


def execute_as_cli(main_file: Path, test_case: TestCase, timeout: int, start_time: float) -> TestResult:
    """Execute test as a CLI subprocess."""
    import time
    
    try:
        # Prepare test inputs
        args, temp_file = prepare_test_input(test_case)
        
        # Build command
        cmd = [sys.executable, str(main_file)] + args
        
        # Execute
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        execution_time = time.time() - start_time
        
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        
        # Get actual output
        actual_output = result.stdout.strip()
        if result.stderr:
            actual_output += f"\nSTDERR: {result.stderr.strip()}"
        
        # Verify output
        passed = verify_output(
            actual_output,
            test_case.expected_output,
            test_case.test_type
        )
        
        return TestResult(
            test_id=test_case.test_id,
            passed=passed,
            actual_output=actual_output,
            expected_output=test_case.expected_output,
            execution_time=execution_time
        )
    
    except subprocess.TimeoutExpired:
        return TestResult(
            test_id=test_case.test_id,
            passed=False,
            actual_output="",
            expected_output=test_case.expected_output,
            error_message=f"Test timed out after {timeout} seconds",
            execution_time=timeout
        )
    
    except Exception as e:
        import time
        return TestResult(
            test_id=test_case.test_id,
            passed=False,
            actual_output="",
            expected_output=test_case.expected_output,
            error_message=f"CLI execution error: {str(e)}",
            execution_time=time.time() - start_time
        )


def verify_output(actual: str, expected: str, test_type: str) -> bool:
    """Verify if actual output matches expected output."""
    
    # For error test cases, check if error message appears
    if test_type == "error":
        # Normalize and check if key error indicators are present
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        # Extract key error terms
        error_terms = ["error", "not found", "invalid", "missing", "failed", "exception"]
        for term in error_terms:
            if term in expected_lower and term in actual_lower:
                return True
        
        # Check for specific expected message parts
        expected_words = expected_lower.split()
        matches = sum(1 for word in expected_words if word in actual_lower)
        return matches >= len(expected_words) * 0.5  # 50% word match
    
    # Normalize outputs
    actual_normalized = actual.strip()
    expected_normalized = expected.strip()
    
    # Try direct string comparison first
    if actual_normalized == expected_normalized:
        return True
    
    # For normal cases, try JSON comparison
    try:
        actual_json = json.loads(actual_normalized)
        expected_json = json.loads(expected_normalized)
        return compare_json_outputs(actual_json, expected_json)
    except json.JSONDecodeError:
        pass
    
    # Try to parse as Python literal (e.g., list representation)
    try:
        import ast
        actual_parsed = ast.literal_eval(actual_normalized)
        expected_parsed = ast.literal_eval(expected_normalized)
        return compare_json_outputs(actual_parsed, expected_parsed)
    except (ValueError, SyntaxError):
        pass
    
    # Try extracting lists/numbers from output strings
    actual_numbers = extract_numbers(actual_normalized)
    expected_numbers = extract_numbers(expected_normalized)
    if actual_numbers and expected_numbers and actual_numbers == expected_numbers:
        return True
    
    # Fall back to string comparison
    return actual_normalized == expected_normalized


def extract_numbers(text: str) -> list:
    """Extract all numbers from a text string."""
    import re
    numbers = re.findall(r'-?\d+\.?\d*', text)
    result = []
    for n in numbers:
        if '.' in n:
            result.append(float(n))
        else:
            result.append(int(n))
    return result


def compare_json_outputs(actual: Any, expected: Any, tolerance: float = 0.01) -> bool:
    """Compare JSON outputs with tolerance for numeric values."""
    
    if type(actual) != type(expected):
        return False
    
    if isinstance(actual, dict):
        if set(actual.keys()) != set(expected.keys()):
            return False
        return all(
            compare_json_outputs(actual[k], expected[k], tolerance)
            for k in actual.keys()
        )
    
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            return False
        return all(
            compare_json_outputs(a, e, tolerance)
            for a, e in zip(actual, expected)
        )
    
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        # Compare numbers with tolerance
        return abs(actual - expected) <= tolerance * max(abs(expected), 1)
    
    else:
        return actual == expected


def run_all_tests(test_cases: list[TestCase], main_file: Optional[Path] = None) -> list[TestResult]:
    """Run all test cases and return results."""
    
    if main_file is None:
        main_file = find_main_file()
    
    if main_file is None:
        print("ERROR: Could not find main Python file to test")
        return []
    
    print(f"Testing file: {main_file}")
    print("=" * 80)
    
    results = []
    for test_case in test_cases:
        print(f"\nRunning {test_case.test_id}: {test_case.description}")
        result = execute_test(main_file, test_case)
        results.append(result)
        
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status} ({result.execution_time:.2f}s)")
        
        if not result.passed:
            print(f"  Expected: {result.expected_output}")
            print(f"  Actual:   {result.actual_output}")
            if result.error_message:
                print(f"  Error:    {result.error_message}")
    
    return results


def print_summary(results: list[TestResult]) -> None:
    """Print test execution summary."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total:  {total}")
    print(f"Passed: {passed} ({100*passed/total:.1f}%)" if total > 0 else "Passed: 0")
    print(f"Failed: {failed}")
    print()
    
    if failed > 0:
        print("Failed tests:")
        for result in results:
            if not result.passed:
                print(f"  - {result.test_id}: {result.error_message or 'Output mismatch'}")


def main() -> None:
    # Example test cases for Fibonacci series
    test_cases = [
        TestCase(
            test_id="TC001",
            description="Fibonacci series with n=10",
            test_type="normal",
            input_data={"n": 10},
            expected_output="[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]",
            validation_criteria="Verify first 10 Fibonacci numbers are returned"
        ),
        TestCase(
            test_id="TC002",
            description="Fibonacci series with n=1",
            test_type="normal",
            input_data={"n": 1},
            expected_output="[0]",
            validation_criteria="Verify only first Fibonacci number is returned"
        ),
        TestCase(
            test_id="TC003",
            description="Fibonacci series with n=0",
            test_type="edge_case",
            input_data={"n": 0},
            expected_output="[]",
            validation_criteria="Verify empty list is returned for n=0"
        ),
        TestCase(
            test_id="TC004",
            description="Fibonacci series with negative n",
            test_type="error",
            input_data={"n": -5},
            expected_output="Error: n must be a non-negative integer",
            validation_criteria="Verify error is raised for negative input"
        ),
        TestCase(
            test_id="TC005",
            description="Fibonacci series with n=5",
            test_type="normal",
            input_data={"n": 5},
            expected_output="[0, 1, 1, 2, 3]",
            validation_criteria="Verify first 5 Fibonacci numbers are returned"
        )
    ]
    
    results = run_all_tests(test_cases)
    print_summary(results)
    
    # Exit with non-zero code if any tests failed
    if any(not r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
