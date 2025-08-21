# generate_pages.py
import importlib.util
from pathlib import Path
import sys
from textwrap import dedent


# === CONFIGURATION ===
PAGE_HEADER = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title} | ⭐ Python Daily Library</title>
  <!-- Open Graph -->
  <meta property="og:title" content="{title}" />
  <meta property="og:description" content="{description}" />
  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://github.com/YOUR_USER/YOUR_REPO/blob/main/scripts/{filename}" target="_blank" />
  <meta property="og:image" content="icon.png" />
  <meta property="og:locale" content="en_US" />

  <!-- Twitter -->
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="{title}" />
  <meta name="twitter:description" content="{description}" />
  
  <link href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism.css" rel="stylesheet" />
  <link href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism.css" rel="stylesheet" />
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/milligram/1.4.1/milligram.min.css">
    {meta}
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" type="image/x-icon" href="favicon.ico">
</head>
<body>
  <header style="padding:1em; background:#f4f4f4;">
    <h1>{title}</h1>
    <p>{description}</p>
  </header>
  <main style="max-width:800px; margin:auto; padding:2em;">
"""

PAGE_FOOTER = """
  </main>
  <footer style="padding:1em; background:#f4f4f4; margin-top:2em;">
    <p>Generated with ❤️ for documentation purposes.</p>
  </footer>
  <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.js"></script>
</body>
</html>
"""

# Config for script paths
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SCRIPTS_DIR = ROOT_DIR / "scripts"
PAGES_DIR = ROOT_DIR / "pages"
PAGES_DIR.mkdir(exist_ok=True)

# =================== FUNCTIONS FOR BUILD PAGES =======================
def load_script_metadata(script_path: Path):
    """Load metadata and function information from a script.

    Args:
        script_path (Path): The path to the script file.

    Returns:
        tuple: A tuple containing the metadata and functions information.
    """
    module_name = f"scripts.{script_path.stem}"
    module = importlib.import_module(module_name)

    metadata = getattr(module, "metadata", None)
    functions = getattr(module, "FUNCTIONS_METADATA", [])

    return metadata, functions



def generate_function_section(func_meta: dict):
    """Generates the HTML for a function as a section (from metadata dictionary)."""
    name = func_meta.get("name", "unknown")
    doc = (func_meta.get("docstring") or "").strip()
    desc = doc.split("\n")[0] if doc else "No description available."

    # Examples information
    examples = ""
    if "Examples:" in doc:
        from textwrap import dedent
        example_block = dedent(doc.split("Examples:")[1]).strip()
        examples = f"""
        <h4>Examples</h4>
        <pre><code class="language-python">{example_block}</code></pre>
        """

    category = func_meta.get("category", "none")
    tags = ", ".join(func_meta.get("tags", []))
    status = func_meta.get("status", "unknown")
    note = func_meta.get("note", "")

    # Build the HTML section for the function
    return f"""
    <section style="margin-bottom:2em;">
      <h2>{name}()</h2>
      <p><strong>Description:</strong> {desc}</p>
      <p><strong>Category:</strong> {category}</p>
      <p><strong>Status:</strong> {status}</p>
      <p><strong>Tags:</strong> {tags or "—"}</p>
      {f"<p><em>Note:</em> {note}</p>" if note else ""}
      {examples}
    </section>
    """

def generate_script_page(script_path: Path, metadata, functions):
    """Generates an HTML page for an individual script."""
    title = metadata.title if metadata else script_path.stem
    description = metadata.description if metadata else ""

    # Links list
    links_html = ""
    if metadata and getattr(metadata, "links", []):
        links_html = "<h2>Links</h2><ul>"
        for link in metadata.links:
            for label, url in link.items():
                links_html += f'<li><a href="{url}" target="_blank">{label}</a></li>'
        links_html += "</ul>"

    # Functions (list of functions decorated with .meta)
    functions_html = "".join(generate_function_section(func_meta) for func_meta in functions)

    page_meta = f"""
    <meta name="description" content="{description}. Script by {metadata.author if metadata else 'unknown author'},
    Version: {metadata.version if metadata else '1'}. Is CLI?: {"✅" if getattr(metadata, "cli", False) else "❌"}">
    """
    filename = f"{script_path.stem}.html"
    page_content = PAGE_HEADER.format(title=title, 
                                      description=description,
                                      meta=page_meta,
                                      filename=filename)
    
    # BUILD
    page_content += f"""
    <h2>General Info 📄</h2>
    <h4>Source</h4>
    <p><a href="https://github.com/YOUR_USER/YOUR_REPO/blob/main/scripts/{script_path.name}" target="_blank">View script on GitHub</a></p>
    <!--
    <ul>
      <li><b>⭐ Version:</b> {metadata.version if metadata else "n/a"}</li>
      <li><b>🧑‍💻 Author:</b> {metadata.author if metadata else "n/a"}</li>
      <li><b>Status:</b> {metadata.status if metadata else "n/a"}</li>
      <li><b>CLI:</b> {"✅" if getattr(metadata, "cli", False) else "❌"}</li>
    </ul> -->

    <h2>Functions ⚙️</h2>
    <hr>
    {functions_html or "<p><i>No functions documented.</i></p>"}

    {links_html}
    """
    page_content += PAGE_FOOTER

    out_path = PAGES_DIR / f"{script_path.stem}.html"
    out_path.write_text(page_content, encoding="utf-8")
    return out_path.name, title


def generate_index(pages_info):
    """Generates an index.html with all scripts."""
    content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Daily Python Library</title>
        <meta property="og:image" content="icon.png" />
        <meta property="og:title" content="Python Daily Library" />
        <meta property="og:description" content="Explore the collection of Python scripts, each with documentation and examples." />
        <meta property="og:type" content="website" />
        <meta property="og:image" content="icon.png" />
        <meta property="og:locale" content="en_US" />

        <!-- Twitter -->
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="Python Daily Library" />
        <meta name="twitter:description" content="Explore the collection of Python scripts, each with documentation and examples." />
        <meta name="twitter:image" content="icon.png" />
        <meta property="og:locale" content="en_US" />
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/milligram/1.4.1/milligram.min.css">
        <link rel="icon" type="image/x-icon" href="icon.png">
      <style>
        body {
          text-align: center;
          padding: 2em;
        }
        main {
          max-width: 800px;
          margin: auto;
        }
        img.hero {
          max-width: 300px;
          margin: 2em auto;
          display: block;
        }
        ol {
          text-align: left;
          margin: 2em auto;
          display: inline-block;
        }
        footer {
          margin-top: 3em;
          font-size: 0.9em;
          opacity: 0.7;
        }
      </style>
    </head>
    <body>
      <main>
        <h1>Daily Python Library</h1>
        <p>Explore a collection of Python scripts from the Daily Python Library, each with documentation and examples.</p>

        <img src="hero.png" alt="Project logo" class="hero">

<p>
          <a href="https://github.com/tu-usuario/daily-python-script" target="_blank" aria-label="GitHub">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 .5C5.648.5.5 5.648.5 12c0 5.088 3.292 9.385 7.865 10.907.575.106.785-.25.785-.556 
                        0-.274-.01-1.002-.015-1.966-3.198.696-3.872-1.54-3.872-1.54-.523-1.328-1.277-1.681-1.277-1.681-1.043-.713.079-.699.079-.699 
                        1.152.081 1.757 1.182 1.757 1.182 1.026 1.757 2.69 1.249 3.345.955.104-.743.402-1.25.73-1.538-2.552-.29-5.238-1.277-5.238-5.68 
                        0-1.255.452-2.281 1.19-3.087-.119-.29-.516-1.456.113-3.037 0 0 .968-.31 3.17 1.177a11.02 11.02 0 0 1 2.884-.388 
                        c.978.004 1.963.132 2.884.388 2.201-1.487 3.168-1.177 3.168-1.177.63 1.581.233 2.747.115 3.037.74.806 1.188 1.832 
                        1.188 3.087 0 4.415-2.69 5.387-5.253 5.671.414.357.782 1.061.782 2.141 0 1.545-.014 2.791-.014 3.171 
                        0 .308.208.667.79.554C20.21 21.38 23.5 17.084 23.5 12c0-6.352-5.148-11.5-11.5-11.5z"/>
            </svg>
            </a>
        </p>
        
        <h2>Available Scripts</h2>
        <ol>
    """
    for filename, title in pages_info:
        content += f'<li><a href="{filename}" target="_blank">{title}</a></li>\n'

    content += """
        </ol>        
      </main>

      <footer>
        <p>Generated by DailyPythonLibrary · © 2025</p>
      </footer>
    </body>
    </html>
    """
    (PAGES_DIR / "index.html").write_text(content, encoding="utf-8")



def main():
    """Generates HTML pages for all Python scripts in the SCRIPTS_DIR."""
    pages_info = []
    for script_path in SCRIPTS_DIR.glob("*.py"):
        if script_path.name == "__init__.py":
            continue
        metadata, functions = load_script_metadata(script_path)
        filename, title = generate_script_page(script_path, metadata, functions)
        pages_info.append((filename, title))

    generate_index(pages_info)
    print("[DONE] Pages generated!")


if __name__ == "__main__":
    main()
