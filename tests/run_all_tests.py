#!/usr/bin/env python
"""
Master Test Runner for AAI Chatbot
Runs all test suites and generates comprehensive report
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


class Colors:
    """ANSI color codes."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class TestRunner:
    """Master test runner."""

    def __init__(self):
        """Initialize test runner."""
        self.test_dir = Path(__file__).parent
        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_test_file(self, test_file: str) -> bool:
        """
        Run a single test file.

        Args:
            test_file (str): Name of test file to run

        Returns:
            bool: True if tests passed
        """
        test_path = self.test_dir / test_file

        if not test_path.exists():
            print(f"{Colors.RED}✗ Test file not found: {test_file}{Colors.RESET}")
            return False

        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
        print(f"Running: {test_file}")
        print(f"{'='*80}{Colors.RESET}\n")

        try:
            result = subprocess.run(
                [sys.executable, str(test_path)],
                capture_output=False,
                timeout=300,
                cwd=str(self.test_dir.parent)
            )

            passed = result.returncode == 0
            self.results[test_file] = {
                "passed": passed,
                "return_code": result.returncode
            }

            return passed

        except subprocess.TimeoutExpired:
            print(f"{Colors.RED}✗ Test timed out: {test_file}{Colors.RESET}")
            self.results[test_file] = {"passed": False, "error": "Timeout"}
            return False
        except Exception as e:
            print(f"{Colors.RED}✗ Error running {test_file}: {e}{Colors.RESET}")
            self.results[test_file] = {"passed": False, "error": str(e)}
            return False

    def run_all_tests(self):
        """Run all test suites."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
        print(f"AAI CHATBOT TEST SUITE")
        print(f"Master Test Runner")
        print(f"{'='*80}{Colors.RESET}\n")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        self.start_time = time.time()

        test_files = [
            "test_new_features.py",
            "test_comprehensive_edge_cases.py",
            "test_scenarios_realistic.py",
            "test_time_aware_comprehensive.py"
        ]

        results = []

        for test_file in test_files:
            try:
                passed = self.run_test_file(test_file)
                results.append((test_file, passed))
            except Exception as e:
                print(f"{Colors.RED}Error running {test_file}: {e}{Colors.RESET}")
                results.append((test_file, False))

        self.end_time = time.time()

        return results

    def print_summary(self, results: list):
        """Print test summary."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        passed_count = sum(1 for _, passed in results if passed)
        failed_count = len(results) - passed_count

        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
        print(f"TEST SUMMARY")
        print(f"{'='*80}{Colors.RESET}\n")

        for test_file, passed in results:
            status = (f"{Colors.GREEN}✓ PASSED{Colors.RESET}" if passed
                     else f"{Colors.RED}✗ FAILED{Colors.RESET}")
            print(f"  {status}: {test_file}")

        print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"Total Tests: {len(results)}")
        print(f"  {Colors.GREEN}Passed: {passed_count}{Colors.RESET}")
        print(f"  {Colors.RED}Failed: {failed_count}{Colors.RESET}")
        print(f"Duration: {duration:.1f}s")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}\n")

        if failed_count == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.RESET}\n")
            return True
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.RESET}\n")
            return False

    def run_interactive_ui_tests(self):
        """Run interactive UI question tests."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
        print(f"Running: test_ui_questions.py (Interactive Mode)")
        print(f"{'='*80}{Colors.RESET}\n")

        test_path = self.test_dir / "test_ui_questions.py"

        if not test_path.exists():
            print(f"{Colors.RED}✗ Test file not found{Colors.RESET}")
            return False

        try:
            subprocess.run(
                [sys.executable, str(test_path)],
                cwd=str(self.test_dir.parent)
            )
            return True
        except Exception as e:
            print(f"{Colors.RED}✗ Error: {e}{Colors.RESET}")
            return False


def main():
    """Main entry point."""
    runner = TestRunner()

    print(f"{Colors.CYAN}Choose test mode:{Colors.RESET}")
    print(f"1. Run All Automated Tests")
    print(f"2. Run Interactive UI Tests")
    print(f"3. Run All (Automated + Interactive)\n")

    mode = input(f"{Colors.CYAN}Select mode (1-3, default=1): "
                f"{Colors.RESET}").strip() or "1"

    if mode == "1":
        results = runner.run_all_tests()
        success = runner.print_summary(results)
        sys.exit(0 if success else 1)

    elif mode == "2":
        runner.run_interactive_ui_tests()

    elif mode == "3":
        results = runner.run_all_tests()
        summary_success = runner.print_summary(results)

        print(f"\n{Colors.YELLOW}Now running interactive UI tests...{Colors.RESET}\n")
        runner.run_interactive_ui_tests()

        sys.exit(0 if summary_success else 1)

    else:
        print(f"{Colors.RED}Invalid mode!{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test runner interrupted{Colors.RESET}")
        sys.exit(1)
