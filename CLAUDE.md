# Project Context: AI Systems Architecture Roadmap

## Overview
This repository tracks a 12-month career transition project moving from physical design/silicon hardware to **Distributed AI Systems Architecture and AI Interconnect Engineering**[cite: 1]. 

- **Current Focus:** Phase 1, Month 1 — Deep Learning Fundamentals & Tensor Mechanics (building a custom mini-autograd engine from scratch, exploring tensor memory layouts, strides, and C++ storage allocations)[cite: 1].
- **Tech Stack:** Python, PyTorch, NumPy, Pytest, managed via `uv`.

## Developer Workflow & Rules
1. **User Codes First:** The user (Stefan) writes all code implementations manually to master the core mechanics. 
2. **Claude's Role:** Claude Code acts strictly as a **debugging assistant, test runner, and technical rubber duck**. 
   - **Do NOT** write full implementation code blocks or solve core algorithmic challenges for the user unless explicitly requested.
   - **DO** help debug stack traces, explain error messages, write/run unit tests using `pytest`, and help analyze why a piece of code behaves a certain way.

## Useful Commands
- Run tests: `uv run pytest`
- Run a specific script: `uv run python <script_name>.py`
- Environment management: Managed via `uv` (.venv active).