# Repository Agent Instructions

Guidelines for agents working in this repository. The goal is to produce focused improvements quickly, with minimal ceremony, while keeping the codebase consistent and safe.

## Scope & Priorities

- Prefer small, surgical patches over broad refactors.
- Fix root causes when practical; avoid unrelated changes.
- Keep style consistent with nearby code; do not add headers or boilerplate.

## What Not To Do

- Do not import external libraries or modules beyond Python’s standard library.
- Do not add or run tests, linters, or other automated checks.
- Do not create ad‑hoc “test” files or scaffolding; propose changes and implement minimal diffs instead.
- Do not modify notebook outputs; when editing `.ipynb`, only touch code and markdown cells.
- Do not replace unicode caharacters with their /uxxx code if not asked explicitely

## Editing & Execution

- Make direct file edits via patch (no need to run local tests here).
- Avoid long‑running commands and heavy dependency changes.
- If a command is necessary to validate behavior, suggest it succinctly for the user to run.

## Documentation & Demos

- Reuse existing code snippets from demo scripts and docs when authoring guides.
- Point users to `PROCESSING_README.md`, `USAGE_GUIDE.md`, and demo files for end‑to‑end workflows.

## Communication

- Summarize intent and the exact files touched.
- When unsure, ask for the expected behavior and constraints before making broad edits.
