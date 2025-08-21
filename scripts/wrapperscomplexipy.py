#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
complexipy.py: Wrappers for complexipy API. Analyze the cognitive complexity of your code.

WITHOUT CLI. complexipy have CLI
"""

import sys
import complexipy as cp
import os

# Allow running as a script or as a module
if __name__ == "__main__" and __package__ is None:
    # Add the parent directory to enable absolute imports
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    __package__ = os.path.basename(os.path.dirname(__file__))

from .scripttypes import ScriptMetadata, function_metadata, FUNCTIONS_METADATA


if cp is None:
    print("complexipy is not installed. Install it with 'pip install complexipy'")
    sys.exit()

metadata = ScriptMetadata(title="Cognitive Complexity Analysis Script",
                           description="How understandable is your codebase? Use complexipy functions to analyze the cognitive complexity of your code.",
                           version="0.1.0",
                           author="Yamil Ayma",
                           email="yamilayma@example.com",
                           license="None",
                           status="development",
                           links=[
                               {"complexipy Documentation": "https://rohaquinlop.github.io/complexipy/#"},
                           ],
                           note="SETUP: pip install complexipy",
                           cli=False)

##########################
#    SCRIPT FUNCTIONS
##########################

@function_metadata(status="development", note="Analyze a single file for cognitive complexity.", category="file_analysis", tags=["complexity", "file"])
def analyze_directory(dir_path: str, threshold:int=15) -> list[str]:
    """Analyze the cognitive complexity of the files in a directory.

    Args:
        dir_path (str): The directory path to analyze.
        threshold (int, optional): The complexity threshold. Defaults to 15.


    Returns:
        list: A report of the cognitive complexity analysis in text format.
    
    Examples:
        >>> analyze_directory(".", threshold=15)
        >>> analyze_directory("src", threshold=20)
    """
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
def compare_complexities(snippet1:str, snippet2:str) -> dict:
    """Compares the cognitive complexity of two code snippets.

    Args:
        snippet1 (str): The first code snippet.
        snippet2 (str): The second code snippet.
    
    Examples:
    >>> compare_complexities("print('Hello, World!')", "print('Hello, Universe!')")
    >>> compare_complexities("def foo(): pass", "def bar(): pass")

    Returns:
        dict: A dictionary with the complexities and comparison result.
    """
    cc1 = cp.code_complexity(snippet1)
    cc2 = cp.code_complexity(snippet2)
    return COMPARISON_STATUSES["BETTER"] if cc2.complexity < cc1.complexity else COMPARISON_STATUSES["WORSE"]


@function_metadata(status="development", note="Generate a CSV report from a Python file's complexity analysis.", category="file_analysis", tags=["complexity", "csv"])
def to_csv(path_file, output_csv="report.csv", row:list[str]=["function", "complexity", "lines"]) -> None:
    """Generates a CSV report from a Python file's complexity analysis.

    Args:
        path_file (str): The path to the Python file to analyze.
        output_csv (str, optional): The path to the output CSV file. Defaults to "report.csv".
        row (list[str], optional): The header row for the CSV file. Defaults to ["function", "complexity", "lines"].

    Examples:
        >>> to_csv("main.py")
    """
    if len(row) != 3:
        raise ValueError("Row must contain exactly 3 elements: example: ['function', 'complexity', 'lines']")
    
    import csv
    fc = cp.file_complexity(path_file)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        for fn in fc.functions:
            writer.writerow([fn.name, fn.complexity, f"{fn.line_start}-{fn.line_end}"])


@function_metadata(status="development", note="Generate a CSV report from a Python file's complexity analysis. You can expand this script with your visualizations.", category="file_analysis", tags=["complexity", "visualization"])
def visualize_complexity_with_matplotlib(path_file:str, threshold:int=15):
    """Visualizes the cognitive complexity of a Python file using a bar chart.

    Args:
        path_file (str): The path to the Python file to analyze.
        threshold (int, optional): The complexity threshold for coloring. Defaults to 15.
        
    Examples:
    >>> visualize_complexity("path/to/your/file.py")
    >>> visualize_complexity("main.py", 30)
        
    """
    import matplotlib.pyplot as plt
    if plt is None:
        print("Matplotlib is not installed, use 'pip install matplotlib'")
        return 
    
    fc = cp.file_complexity(path_file)
    functions = [fn.name for fn in fc.functions]
    complexities = [fn.complexity for fn in fc.functions]

    # Colors: green if <= threshold, red if > threshold
    colors = ['green' if c <= threshold else 'red' for c in complexities]
    
    plt.figure(figsize=(10, 6))
    plt.barh(functions, complexities, color=colors)
    plt.axvline(threshold, color='orange', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Cognitive Complexity')
    plt.ylabel('Functions')
    plt.title(f'Complexity by Function in {fc.file_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('complexity_chart.png')  # Save as image
    plt.show()  # Display on screen
    print("Chart generated and saved as 'complexity_chart.png'.")



STATUS_GIT_COMPLEXITY = {
    "BETTER": 0,
    "WORSE": 1,
    "NO_CHANGE": 2
}

@function_metadata(status="development", note="Generate a report from a Python file's complexity analysis with comparison to previous commits.", category="file_analysis", tags=["complexity", "git"])
def compare_complexity_git(path_file, before_commit='HEAD~1'):
    """Compares the complexity of a Python file between the current state and a previous Git commit.

    Args:
        path_file (str): The path to the Python file to analyze.
        before_commit (str, optional): The Git commit to compare against. Defaults to 'HEAD~1'.

    Returns:
        dict: A dictionary with the complexities and comparison result.
        
    Examples:
        >>> compare_complexity_git("main.py", before_commit='HEAD~2')
        >>> compare_complexity_git("main.py", before_commit='HEAD~3')
    """
    import subprocess
    
    # Get currently code
    with open(path_file, 'r') as f:
        currently_code = f.read()
    cc_actual = cp.code_complexity(currently_code)

    # Get previous commit code via Git
    try:
        before_code = subprocess.check_output(
            ['git', 'show', f'{before_commit}:{path_file}'],
            text=True
        )
        cc_anterior = cp.code_complexity(before_code)

        difference = cc_actual.complexity - cc_anterior.complexity
        status = STATUS_GIT_COMPLEXITY["BETTER"] if difference < 0 else STATUS_GIT_COMPLEXITY["WORSE"] if difference > 0 else STATUS_GIT_COMPLEXITY["NO_CHANGE"]

        print(f"Complexity anterior: {cc_anterior.complexity}")
        print(f"Complexity actual: {cc_actual.complexity}")
        print(f"Difference: {difference} ({status})")

        return difference
    except subprocess.CalledProcessError:
        print("Error: Commit not found")
        return None





 ##########################
#    CLI FUNCTIONS
###########################


##########################
#    EXECUTE
##########################


if __name__ == "__main__":
    print("This script is not intended to be run directly. Use the functions defined in your codebase.")
    pass
""" 
Author: Yamil Ayma
Date: 14/08/2025
From: DailyPythonLibraries
"""
