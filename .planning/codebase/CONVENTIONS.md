# Coding Conventions

- **Package layout:** Feature-oriented subpackages under `src/` (`ingestion`, `retrieval`, `generation`, `utils`, `database`) imported via absolute package paths (`from src.<module> import ...`).
- **Naming:** `snake_case` for functions, variables, and modules; `UPPER_SNAKE_CASE` for constants in `src/config.py` and `src/generation/prompts.py`.
- **Error handling:** Raise `ValueError` for invalid inputs or missing preconditions (e.g., unreadable PDF, missing index, missing API key); catch and render via `st.error(...)` at the `app.py` UI boundary.
- **Prompts:** Keep LLM system and context formatting templates centralized in `src/generation/prompts.py`.
- **Secrets:** Read exclusively through `src/config.py` via `python-dotenv`; never hardcode keys.
