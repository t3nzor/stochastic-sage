# AGENTS.md - Agentic Coding Guidelines for stochastic-sage

## Project Overview
This is a Python project - a Gradio-based chatbot using Qwen3.5-9B model. It uses `gradio`, `transformers`, `accelerate`, and `torch`.

## Build/Lint/Test Commands

### Installation
```bash
pip install gradio transformers accelerate torch
```

### Running the Application
```bash
python chatbot.py
```
The chatbot will launch at `http://localhost:7860`.

### Testing
This project currently has **no tests**. If adding tests:

```bash
# Run all tests with pytest
pytest

# Run a single test file
pytest tests/test_chatbot.py

# Run a single test function
pytest tests/test_chatbot.py::test_chat_function

# Run tests matching a pattern
pytest -k "test_chat"

# Run with coverage
pytest --cov=. --cov-report=html
```

### Linting (recommended additions)
If you add linting, install `ruff`:
```bash
pip install ruff
```

```bash
# Run linter
ruff check .

# Run formatter
ruff format .

# Fix auto-fixable issues
ruff check . --fix
```

### Type Checking (recommended)
```bash
pip install mypy
mypy .
```

---

## Code Style Guidelines

### General Principles
- Write clean, readable code over clever code
- Keep functions focused and small (ideally <50 lines)
- Use descriptive variable names
- Avoid magic numbers - use named constants

### Formatting
- Use **4 spaces** for indentation (not tabs)
- Maximum line length: **100 characters**
- Add blank lines between functions and classes
- Use Black-compatible formatting if using formatters

### Imports
- Group imports in this order: standard library, third-party, local
- Use absolute imports for packages
- Avoid wildcard imports (`from x import *`)
- Example:
  ```python
  import random
  import sys
  
  import gradio as gr
  import torch
  from transformers import AutoTokenizer, AutoModelForCausalLM
  
  from . import local_module
  ```

### Naming Conventions
- **Functions/variables**: `snake_case` (e.g., `def chat(user_input):`)
- **Classes**: `PascalCase` (e.g., `class ChatBot:`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MODEL_ID`)
- **Private variables**: prefix with underscore (e.g., `_private_var`)

### Type Hints
Use type hints where beneficial:
```python
def chat(user_input: str, messages: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Process user input and return updated message history."""
    if messages is None:
        messages = []
    # ...
    return messages
```

### Error Handling
- Use specific exception types
- Include meaningful error messages
- Handle exceptions at appropriate levels
- Example:
  ```python
  try:
      result = risky_operation()
  except SpecificError as e:
      logger.error(f"Operation failed: {e}")
      raise CustomError("Fallback message") from e
  ```

### Docstrings
Use Google-style docstrings:
```python
def function(param1: str, param2: int) -> bool:
    """Short one-line description.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param2 is negative.
    """
```

### Comments
- Write comments that explain **why**, not **what**
- Keep comments up-to-date with code changes
- Use TODO comments for future work: `# TODO(username): description`

### Testing Guidelines
- Test one thing per test function
- Use descriptive test names: `test_chat_returns_message_history()`
- Follow AAA pattern: Arrange, Act, Assert
- Mock external dependencies (API calls, file I/O)

### Git Conventions
- Write concise commit messages (50 chars max for subject)
- Use imperative mood: "Add feature" not "Added feature"
- Reference issues in commits: "Fix #123: resolve bug"

### File Organization
```
project/
├── chatbot.py          # Main application
├── tests/              # Test files
│   └── test_chatbot.py
├── pyproject.toml      # Project config
├── ruff.toml           # Linter config
└── .env                # Environment variables (never commit)
```

---

## Recommendations for Future Development

1. **Add pyproject.toml** for project metadata and dependency management
2. **Add ruff.toml** for consistent linting/formatting
3. **Add type hints** throughout the codebase
4. **Add tests** before adding features
5. **Use environment variables** for configuration (model paths, API keys)
6. **Add CI/CD** (GitHub Actions) for automated testing and linting
