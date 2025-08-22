
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import inspect
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

def import_module(path: Path):
    module_name = f"scripts.{path.stem}"
    spec = spec_from_file_location(module_name, path)
    mod = module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def get_decorated_functions(module, ignore=("main",)):
    return [
        (name, func)
        for name, func in inspect.getmembers(module, inspect.isfunction)
        if hasattr(func, "meta") and name not in ignore
    ]

def generate_script_section(number:int, script_path: Path) -> str:
    try:
        module = import_module(script_path)
    except Exception as e:
        # return f"\n<!-- ⚠️ Error loading {script_path.name}: {e} -->\n"
        return ""
    
    metadata = getattr(module, "metadata", None)
    if not metadata:
        # return f"\n<!-- ⚠️ No metadata in {script_path.name} -->\n"
        return "<!-- ⚠️ No metadata found -->7"
    # Header
    general_info = f"""| Description | CLI |
|-------------|---------|
| {metadata.description} | {"✅" if metadata.cli else "❌"} |\n
"""

    # All python functions
    functions = get_decorated_functions(module)
    if functions:
        table = "| Function | Description | Category | Tags | Status |\n"
        table += "|----------|-------------|----------|------|--------|\n"
        for name, func in functions:
            meta = func.meta
            desc = (func.__doc__ or "").strip().split('\n')[0]
            table += f"| `{name}()` | {desc} | {meta['category']} | {', '.join(meta['tags'])} | {meta['status']} |\n"
    else:
        table = "*No documented functions.*"

    # Note
    note = f"\n📝 **Note**: {metadata.note}" if metadata.note else ""
    
    # Links section 
    if getattr(metadata, "links", None):
        links_table = "| Name | URL |\n|------|-----|\n"
        for link in metadata.links:
            for name, url in link.items():
                links_table += f"| {name} | [{url}]({url}) |\n"
    else:
        links_table = "*No external links provided.*"

    footer_info = f"""
🔗 **Links**
{links_table}
    """

    return f"""<details>
<summary><a href="scripts/{script_path.name}">{number}. 📘 {script_path.name}</a> <kbd>Show details</kbd></summary>

{general_info}
{table}
{note}
{footer_info}
---

</details>
"""

def generate_all_sections():
    ignore_files = ["__init__.py", "scripttypes.py"]
    output = "# 🧩 Scripts\n\n"
    scripts = list(SCRIPTS_DIR.glob("*.py"))

    # For generate scripts sections
    i = 1
    for script in scripts:
        if script.name in ignore_files:
            continue
        output += generate_script_section(i, script) + "\n"
        i += 1
    return output
