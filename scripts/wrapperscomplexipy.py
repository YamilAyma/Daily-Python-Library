#!/usr/bin/env python3
# -*- a: utf-8 -*-
"""
complexipy.py: Wrappers for complexipy API. Analyze the cognitive complexity of your code.

WITHOUT CLI. complexipy have CLI
"""

import sys
import os

# Allow running as a script or as a module
if __name__ == "__main__" and __package__ is None:
    # Add the parent directory to enable absolute imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

from .scripttypes import ScriptMetadata, function_metadata, FUNCTIONS_METADATA
FUNCTIONS_METADATA.clear()  # Don't remove, it's for documentation


metadata = ScriptMetadata(
    title="Cognitive Complexity Analysis Script",
    description="How understandable is your codebase? Use complexipy functions to analyze the cognitive complexity of your code.",
    version="0.2.0", # Version bump for lazy loading refactor
    author="Yamil Ayma",
    email="yamilayma@example.com",
    license="None",
    status="development",
    dependencies=["complexipy", "matplotlib"],
    links=[
        {"complexipy Documentation": "https://rohaquinlop.github.io/complexipy/#"},
    ],
    note="SETUP: pip install complexipy matplotlib",
    cli=False
)

##########################
#    SCRIPT FUNCTIONS
##########################

@function_metadata(status="development", note="Analyze a single file for cognitive complexity.", category="file_analysis", tags=["complexity", "file"])
def analyze_directory(dir_path: str, threshold: int = 15) -> list[str]:
    """
    Analyze the cognitive complexity of the files in a directory.

    Args:
        dir_path (str): The directory path to analyze.
        threshold (int, optional): The complexity threshold. Defaults to 15.

    Returns:
        list: A report of the cognitive complexity analysis in text format.
    
    Examples:
        >>> analyze_directory(".", threshold=15)
    """
    try:
        import complexipy as cp
    except ImportError as e:
        raise ImportError("complexipy is not installed. Please run: 'pip install complexipy'") from e

    report = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                fc = cp.file_complexity(path)
                status = "OK" if fc.complexity <= threshold else "ALERT"
                report.append(f"{fc.file_name}: Complexity {fc.complexity} ({status})")
    return report

COMPARISON_STATUSES = {
    "BETTER": 0,
    "WORSE": 1
}

@function_metadata(status="development", note="Analyze a code snippet for cognitive complexity.", category="code_analysis", tags=["complexity", "code"])
def compare_complexities(snippet1: str, snippet2: str) -> int:
    """
    Compares the cognitive complexity of two code snippets.

    Args:
        snippet1 (str): The first code snippet.
        snippet2 (str): The second code snippet.
    
    Returns:
        int: A status code indicating if snippet2 is better (0) or worse (1).
    """
    try:
        import complexipy as cp
    except ImportError as e:
        raise ImportError("complexipy is not installed. Please run: 'pip install complexipy'") from e
        
    cc1 = cp.code_complexity(snippet1)
    cc2 = cp.code_complexity(snippet2)
    return COMPARISON_STATUSES["BETTER"] if cc2.complexity < cc1.complexity else COMPARISON_STATUSES["WORSE"]


@function_metadata(status="development", note="Generate a CSV report from a Python file's complexity analysis.", category="file_analysis", tags=["complexity", "csv"])
def to_csv(path_file: str, output_csv: str = "report.csv", row: list[str] = None) -> None:
    """
    Generates a CSV report from a Python file's complexity analysis.

    Args:
        path_file (str): The path to the Python file to analyze.
        output_csv (str, optional): The path to the output CSV file. Defaults to "report.csv".
        row (list[str], optional): The header row for the CSV. Defaults to ["function", "complexity", "lines"].
    """
    try:
        import complexipy as cp
    except ImportError as e:
        raise ImportError("complexipy is not installed. Please run: 'pip install complexipy'") from e
    import csv

    if row is None:
        row = ["function", "complexity", "lines"]
    if len(row) != 3:
        raise ValueError("Row must contain exactly 3 elements: e.g., ['function', 'complexity', 'lines']")
    
    fc = cp.file_complexity(path_file)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        for fn in fc.functions:
            writer.writerow([fn.name, fn.complexity, f"{fn.line_start}-{fn.line_end}"])
    print(f"Complexity report saved to '{output_csv}'.")


@function_metadata(status="development", note="Visualize a file's complexity analysis. You can expand this script with your visualizations.", category="file_analysis", tags=["complexity", "visualization"])
def visualize_complexity_with_matplotlib(path_file: str, threshold: int = 15):
    """
    Visualizes the cognitive complexity of a Python file using a bar chart.

    Args:
        path_file (str): The path to the Python file to analyze.
        threshold (int, optional): The complexity threshold for coloring. Defaults to 15.
    """
    try:
        import matplotlib.pyplot as plt
        import complexipy as cp
    except ImportError:
        print("Error: Required libraries are not installed. Please run: 'pip install complexipy matplotlib'")
        return 
    
    fc = cp.file_complexity(path_file)
    functions = [fn.name for fn in fc.functions]
    complexities = [fn.complexity for fn in fc.functions]

    colors = ['green' if c <= threshold else 'red' for c in complexities]
    
    plt.figure(figsize=(10, 6))
    plt.barh(functions, complexities, color=colors)
    plt.axvline(threshold, color='orange', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Cognitive Complexity')
    plt.ylabel('Functions')
    plt.title(f'Complexity by Function in {fc.file_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('complexity_chart.png')
    plt.show()
    print("Chart generated and saved as 'complexity_chart.png'.")

STATUS_GIT_COMPLEXITY = {
    "BETTER": 0,
    "WORSE": 1,
    "NO_CHANGE": 2
}

@function_metadata(status="development", note="Compare a file's complexity against a previous Git commit.", category="file_analysis", tags=["complexity", "git"])
def compare_complexity_git(path_file: str, before_commit: str = 'HEAD~1') -> int | None:
    """
    Compares the complexity of a Python file between the current state and a previous Git commit.

    Args:
        path_file (str): The path to the Python file to analyze.
        before_commit (str, optional): The Git commit to compare against. Defaults to 'HEAD~1'.

    Returns:
        int | None: The difference in complexity, or None if the commit is not found.
    """
    try:
        import complexipy as cp
    except ImportError as e:
        raise ImportError("complexipy is not installed. Please run: 'pip install complexipy'") from e
    import subprocess
    
    with open(path_file, 'r') as f:
        currently_code = f.read()
    cc_actual = cp.code_complexity(currently_code)

    try:
        before_code = subprocess.check_output(
            ['git', 'show', f'{before_commit}:{path_file}'],
            text=True,
            stderr=subprocess.PIPE
        )
        cc_anterior = cp.code_complexity(before_code)

        difference = cc_actual.complexity - cc_anterior.complexity
        status_key = "BETTER" if difference < 0 else "WORSE" if difference > 0 else "NO_CHANGE"
        status = STATUS_GIT_COMPLEXITY[status_key]

        print(f"Previous complexity ({before_commit}): {cc_anterior.complexity}")
        print(f"Current complexity: {cc_actual.complexity}")
        print(f"Difference: {difference} (Status: {status_key})")

        return difference
    except subprocess.CalledProcessError:
        print(f"Error: Could not find file '{path_file}' in commit '{before_commit}'.")
        return None

##########################
#    EXECUTE
##########################

if __name__ == "__main__":
    print("This script is not intended to be run directly. Use the functions defined in your codebase.")