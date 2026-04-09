"""
Integrated Code Synthesis Pipeline

This module integrates all components of the code synthesis system:
1. Prompt Processor - Extracts problem specification from user input
2. Code Generator - Generates code from specification
3. Test Case Generator - Creates test cases from specification
4. Test Executor - Runs tests against generated code
5. Error Analyzer - Analyzes failed tests
6. Code Repair - Fixes code based on error analysis

The pipeline supports iterative refinement until all tests pass or max iterations reached.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional

ProgressCallback = Optional[Callable[[str, str, str, float], None]]

# Import all modules
from prompt_processor import process_user_input, ProblemSpecification
from code_generator import (
    generate_code,
    write_generated_code_to_file,
    GeneratedCodeFile,
)
from test_case_generator import generate_test_cases, TestCaseCollection
from test_execution_verify import (
    run_all_tests,
    print_summary,
    TestResult,
    TestCase as ExecutionTestCase,
    find_main_file,
)
from error_analyser import analyze_errors
from repair_prompt import repair_code
from memory import GenerationMemory


@dataclass
class PipelineConfig:
    """Configuration for the synthesis pipeline."""

    output_dir: str = "real"
    max_repair_iterations: int = 2
    acceptance_threshold: float = 0.85
    model: str = "gemma4:e4b"
    verbose: bool = True


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""

    success: bool
    problem_spec: Optional[ProblemSpecification] = None
    generated_code: Optional[GeneratedCodeFile] = None
    test_cases: Optional[TestCaseCollection] = None
    test_results: list[TestResult] = field(default_factory=list)
    repair_iterations: int = 0
    final_code: Optional[str] = None
    error_analysis: Optional[dict] = None
    output_file: Optional[str] = None
    accepted: bool = False
    all_tests_passed: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    pass_rate: float = 0.0
    acceptance_threshold: float = 0.85


def log(message: str, verbose: bool = True):
    """Print message if verbose mode is enabled."""
    if verbose:
        print(message)


def step_1_process_prompt(
    user_input: str, config: PipelineConfig
) -> ProblemSpecification:
    """Step 1: Process user input into structured problem specification."""
    log("\n" + "=" * 80, config.verbose)
    log("STEP 1: Processing User Input", config.verbose)
    log("=" * 80, config.verbose)

    spec = process_user_input(user_input)

    log(f"\nProblem Summary: {spec.problem_summary}", config.verbose)
    log(f"Inputs: {spec.inputs}", config.verbose)
    log(f"Outputs: {spec.outputs}", config.verbose)
    log(f"Constraints: {spec.constraints}", config.verbose)
    log(f"Edge Cases: {spec.edge_cases}", config.verbose)
    log(f"Assumptions: {spec.assumptions}", config.verbose)

    return spec


def step_2_generate_code(
    spec: ProblemSpecification, config: PipelineConfig
) -> GeneratedCodeFile:
    """Step 2: Generate code from problem specification."""
    log("\n" + "=" * 80, config.verbose)
    log("STEP 2: Generating Code", config.verbose)
    log("=" * 80, config.verbose)

    generated = generate_code(spec.model_dump())

    log(f"\nGenerated file: {generated.file_name}", config.verbose)
    log(f"Code length: {len(generated.code)} characters", config.verbose)

    return generated


def step_3_generate_tests(
    spec: ProblemSpecification, config: PipelineConfig, generated_code: str
) -> TestCaseCollection:
    """Step 3: Generate test cases from problem specification AND generated code."""
    log("\n" + "=" * 80, config.verbose)
    log("STEP 3: Generating Test Cases", config.verbose)
    log("=" * 80, config.verbose)

    test_cases = generate_test_cases(
        spec, generated_code=generated_code, model_name=config.model
    )

    log(f"\nGenerated {len(test_cases.test_cases)} test cases:", config.verbose)
    for tc in test_cases.test_cases:
        log(f"  - {tc.test_id}: {tc.description} ({tc.test_type})", config.verbose)

    return test_cases


def step_4_execute_tests(
    code: GeneratedCodeFile, test_cases: TestCaseCollection, config: PipelineConfig
) -> list[TestResult]:
    """Step 4: Execute tests against generated code."""
    log("\n" + "=" * 80, config.verbose)
    log("STEP 4: Executing Tests", config.verbose)
    log("=" * 80, config.verbose)

    # Write code to file
    output_path = write_generated_code_to_file(
        code, out_dir=config.output_dir, overwrite=True
    )
    log(f"\nCode written to: {output_path}", config.verbose)

    # Find the main file
    main_file = find_main_file(config.output_dir)
    if main_file is None:
        log("ERROR: Could not find main file to test", config.verbose)
        return []

    # Run tests
    executable_tests = [
        ExecutionTestCase.model_validate(tc.model_dump())
        for tc in test_cases.test_cases
    ]
    results = run_all_tests(executable_tests, main_file)

    if config.verbose:
        print_summary(results)

    return results


def step_5_analyze_errors(
    code: str, failed_results: list[TestResult], config: PipelineConfig
) -> dict:
    """Step 5: Analyze errors from failed tests."""
    log("\n" + "=" * 80, config.verbose)
    log("STEP 5: Analyzing Errors", config.verbose)
    log("=" * 80, config.verbose)

    # Convert failed results to dict format for error analyzer
    failed_tests = [
        {
            "test_id": r.test_id,
            "expected": r.expected_output,
            "actual": r.actual_output,
            "error": r.error_message or "Output mismatch",
        }
        for r in failed_results
    ]

    analysis = analyze_errors(code, failed_tests, model=config.model)

    log(f"\nError Summary: {analysis.get('error_summary', 'N/A')}", config.verbose)
    log(f"Error Categories: {analysis.get('error_categories', [])}", config.verbose)
    log(f"Root Causes: {analysis.get('root_causes', [])}", config.verbose)

    return analysis


def step_6_repair_code(code: str, error_analysis: dict, config: PipelineConfig) -> str:
    """Step 6: Repair code based on error analysis."""
    log("\n" + "=" * 80, config.verbose)
    log("STEP 6: Repairing Code", config.verbose)
    log("=" * 80, config.verbose)

    repair_result = repair_code(code, error_analysis, model=config.model)
    repaired_code = repair_result.get("repaired_code", code)

    log(f"\nRepaired code length: {len(repaired_code)} characters", config.verbose)

    return repaired_code


def _emit(
    callback: ProgressCallback, step: str, status: str, message: str, progress: float
):
    if callback is not None:
        callback(step, status, message, progress)


def run_pipeline(
    user_input: str,
    config: Optional[PipelineConfig] = None,
    on_progress: ProgressCallback = None,
) -> PipelineResult:
    """
    Run the complete code synthesis pipeline.

    Args:
        user_input: Natural language description of the desired program
        config: Pipeline configuration (optional)

    Returns:
        PipelineResult with all outputs and status
    """
    if config is None:
        config = PipelineConfig()

    result = PipelineResult(success=False)
    result.acceptance_threshold = config.acceptance_threshold
    memory = GenerationMemory()

    try:
        # Step 1: Process prompt
        _emit(
            on_progress,
            "processing_prompt",
            "running",
            "Processing user prompt...",
            0.0,
        )
        result.problem_spec = step_1_process_prompt(user_input, config)
        _emit(on_progress, "processing_prompt", "completed", "Prompt processed", 0.15)

        # Step 2: Generate code
        _emit(
            on_progress,
            "generating_code",
            "running",
            "Generating code from specification...",
            0.15,
        )
        result.generated_code = step_2_generate_code(result.problem_spec, config)
        current_code = result.generated_code.code
        current_file = result.generated_code
        _emit(on_progress, "generating_code", "completed", "Code generated", 0.35)

        # Step 3: Generate test cases
        _emit(
            on_progress, "generating_tests", "running", "Generating test cases...", 0.35
        )
        result.test_cases = step_3_generate_tests(
            result.problem_spec, config, result.generated_code.code
        )
        _emit(
            on_progress,
            "generating_tests",
            "completed",
            f"Generated {len(result.test_cases.test_cases)} test cases",
            0.50,
        )

        # Iterative testing and repair loop
        for iteration in range(config.max_repair_iterations + 1):
            log(f"\n{'#' * 80}", config.verbose)
            log(f"ITERATION {iteration + 1}", config.verbose)
            log(f"{'#' * 80}", config.verbose)

            # Step 4: Execute tests
            iter_label = f" (iteration {iteration + 1})" if iteration > 0 else ""
            _emit(
                on_progress,
                "executing_tests",
                "running",
                f"Running tests{iter_label}...",
                0.50,
            )
            result.test_results = step_4_execute_tests(
                current_file, result.test_cases, config
            )

            # Check if all tests passed
            failed_results = [r for r in result.test_results if not r.passed]
            passed = len(result.test_results) - len(failed_results)
            total = len(result.test_results)

            label = "initial" if iteration == 0 else f"repair_{iteration}"
            memory.add(current_code, passed, total, label=label)

            pass_rate = (passed / total) if total > 0 else 0.0

            result.tests_passed = passed
            result.tests_total = total
            result.pass_rate = pass_rate

            if total > 0 and pass_rate >= config.acceptance_threshold:
                if passed == total:
                    log("\n✓ All tests passed!", config.verbose)
                    status_message = f"All {total} tests passed!"
                else:
                    pct = round(pass_rate * 100, 1)
                    log(
                        f"\n✓ Acceptance threshold met ({passed}/{total}, {pct}%)",
                        config.verbose,
                    )
                    status_message = f"Accepted: {passed}/{total} tests passed ({pct}%)"
                _emit(
                    on_progress,
                    "executing_tests",
                    "completed",
                    status_message,
                    0.95,
                )
                result.accepted = True
                result.all_tests_passed = passed == total
                result.success = True
                result.final_code = current_code
                result.repair_iterations = iteration
                break

            log(f"\n✗ {len(failed_results)} test(s) failed", config.verbose)
            _emit(
                on_progress,
                "executing_tests",
                "completed",
                f"{passed}/{total} tests passed, {len(failed_results)} failed",
                0.70,
            )

            # Don't repair on the last iteration
            if iteration >= config.max_repair_iterations:
                log(
                    f"\nMax repair iterations ({config.max_repair_iterations}) reached",
                    config.verbose,
                )
                result.final_code = current_code
                result.repair_iterations = iteration
                break

            # Step 5: Analyze errors
            _emit(
                on_progress,
                "analyzing_errors",
                "running",
                f"Analyzing {len(failed_results)} failed test(s){iter_label}...",
                0.70,
            )
            result.error_analysis = step_5_analyze_errors(
                current_code, failed_results, config
            )
            _emit(
                on_progress,
                "analyzing_errors",
                "completed",
                "Error analysis complete",
                0.85,
            )

            # Step 6: Repair code
            _emit(
                on_progress,
                "repairing_code",
                "running",
                f"Repairing code{iter_label}...",
                0.85,
            )
            repaired_code = step_6_repair_code(
                current_code, result.error_analysis, config
            )
            _emit(
                on_progress,
                "repairing_code",
                "completed",
                "Code repaired, re-testing...",
                0.95,
            )

            # Quick test of repaired code to decide rollback
            repaired_file = GeneratedCodeFile(
                file_name=result.generated_code.file_name, code=repaired_code
            )
            repaired_results = step_4_execute_tests(
                repaired_file, result.test_cases, config
            )
            repaired_passed = sum(1 for r in repaired_results if r.passed)
            repaired_total = len(repaired_results)

            if memory.should_rollback(repaired_passed, repaired_total):
                best = memory.best
                if best is not None and result.generated_code is not None:
                    log(
                        f"\n⚠ Repair degraded quality ({repaired_passed}/{repaired_total} vs best {best.test_passed}/{best.test_total}), rolling back...",
                        config.verbose,
                    )
                    current_code = best.code
                    current_file = GeneratedCodeFile(
                        file_name=result.generated_code.file_name, code=best.code
                    )
                else:
                    current_code = repaired_code
                    current_file = repaired_file
            else:
                current_code = repaired_code
                current_file = repaired_file

        # Write final code to output
        _emit(on_progress, "saving_output", "running", "Saving final code...", 0.95)
        result.output_file = write_generated_code_to_file(
            GeneratedCodeFile(
                file_name=result.generated_code.file_name,
                code=result.final_code or current_code,
            ),
            out_dir=config.output_dir,
            overwrite=True,
        )
        _emit(on_progress, "completed", "completed", "Pipeline finished", 1.0)

    except Exception as e:
        _emit(on_progress, "failed", "failed", str(e), 0.0)
        log(f"\nERROR: Pipeline failed with exception: {str(e)}", config.verbose)
        import traceback

        traceback.print_exc()

    return result


def print_pipeline_result(result: PipelineResult):
    """Print a summary of the pipeline result."""
    print("\n" + "=" * 80)
    print("PIPELINE RESULT SUMMARY")
    print("=" * 80)

    if result.all_tests_passed:
        summary_status = "SUCCESS ✓"
    elif result.accepted:
        threshold_pct = round(result.acceptance_threshold * 100, 1)
        summary_status = f"ACCEPTED (>={threshold_pct}%) ✓"
    else:
        summary_status = "FAILED ✗"

    print(f"\nStatus: {summary_status}")
    print(f"Repair iterations: {result.repair_iterations}")

    if result.test_results:
        passed = sum(1 for r in result.test_results if r.passed)
        total = len(result.test_results)
        print(f"Tests: {passed}/{total} passed")

    if result.output_file:
        print(f"Output file: {result.output_file}")

    if result.error_analysis and not result.success:
        print(f"\nLast Error Analysis:")
        print(f"  Summary: {result.error_analysis.get('error_summary', 'N/A')}")
        print(f"  Root Causes: {result.error_analysis.get('root_causes', [])}")


def main():
    """Interactive usage of the integrated pipeline."""
    import sys

    # Get user input - either from command line or interactively
    if len(sys.argv) > 1:
        # User provided input as command line argument
        user_input = " ".join(sys.argv[1:])
    else:
        # Interactive mode - ask user for input
        print("=" * 80)
        print("CODE SYNTHESIS PIPELINE")
        print("=" * 80)
        print("\nDescribe the program you want to create:")
        user_input = input("> ").strip()

        if not user_input:
            print("No input provided. Exiting.")
            return

    print(f"\nUser Request: {user_input}")

    # Configure the pipeline
    config = PipelineConfig(
        output_dir="real", max_repair_iterations=2, model="gemma4:e4b", verbose=True
    )

    # Run the pipeline
    result = run_pipeline(user_input, config)

    # Print summary
    print_pipeline_result(result)

    # Show final code
    if result.final_code:
        print("\n" + "=" * 80)
        print("FINAL CODE")
        print("=" * 80)
        print(result.final_code)


if __name__ == "__main__":
    main()
