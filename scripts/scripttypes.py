from functools import wraps

FUNCTIONS_METADATA = []

class ScriptMetadata:
    """
    Lightweight utility for managing script metadata.
    Compatible with CLI, automated workflows, and data export.
    """

    _statuses = {
        "development": "Development",
        "production": "Production",
        "testing": "Testing"
    }

    def __init__(self, *, title, description, version, author, email,
                 license="MIT", status="development", links=None,
                 note=None, cli=True, dependencies=None, tags=None):
        self.title = title
        self.description = description
        self.version = version
        self.author = author
        self.email = email
        self.license = license
        self.status = self._statuses.get(status, "Unknown")
        self.links = links or []
        self.note = note or ""
        self.cli = cli
        self.dependencies = dependencies or []
        self.tags = tags or []

    def to_dict(self):
        """Returns the metadata as a dictionary (ideal for YAML/JSON)."""
        return {
            "title": self.title,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "email": self.email,
            "license": self.license,
            "status": self.status,
            "cli": self.cli,
            "note": self.note,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "links": self.links
        }

    def to_table_row(self):
        """Returns an ordered list of the main fields (for Markdown tables)."""
        return [
            self.title,
            self.description,
            self.version,
            self.status,
            self.license,
            ", ".join(self.dependencies),
            "✅" if self.cli else "❌"
        ]

    def print(self):
        """Show metadata in a user-friendly way in the terminal."""
        print("=" * 50)
        print(f"📌 {self.title}")
        print(f"{self.description}")
        print(f"🔖 Version: {self.version}")
        print(f"🧑 Author: {self.author} ({self.email})")
        print(f"🛡️ License: {self.license}")
        print(f"🚦 Status: {self.status}")
        if self.dependencies:
            print(f"📦 Dependencies: {', '.join(self.dependencies)}")
        if self.tags:
            print(f"🏷️ Tags: {', '.join(self.tags)}")
        if self.note:
            print(f"📝 Note: {self.note}")
        if self.links:
            print("🔗 Links:")
            for link in self.links:
                print(f"  - {link}")
        print("=" * 50)



def function_metadata(category=None, tags=None, status=None, note=None):
    """
    Adds additional metadata to a function and registers it globally.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        metadata = {
            "name": func.__name__,
            "docstring": func.__doc__ or "",
            "category": category or "",
            "tags": tags or [],
            "status": status or "unknown",
            "note": note or "",
        }

        wrapper.meta = metadata

        FUNCTIONS_METADATA.append(metadata)

        return wrapper
    return decorator

