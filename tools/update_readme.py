
from generate_readme import generate_all_sections
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
README_PATH = Path(__file__).resolve().parent.parent / "README.md"
START_TAG = "<!-- SCRIPT_TABLE_START -->"
END_TAG = "<!-- SCRIPT_TABLE_END -->"

def update_readme():
    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    if START_TAG not in content or END_TAG not in content:
        raise Exception("Not found tags in README.md")

    before = content.split(START_TAG)[0]
    after = content.split(END_TAG)[1]
    new_section = generate_all_sections()

    updated = f"{before}{START_TAG}\n{new_section}\n{END_TAG}{after}"

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(updated)

    print("[✓] README updated.")

if __name__ == "__main__":
    update_readme()
