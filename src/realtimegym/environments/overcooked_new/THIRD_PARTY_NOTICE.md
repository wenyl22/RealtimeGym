# Third-Party Code Notice

This directory (`overcooked_new`) contains vendored code from the **Overcooked-AI** project.

## Source

- **Project**: Overcooked-AI
- **Repository**: https://github.com/HumanCompatibleAI/overcooked_ai
- **Authors**: Nathan Miller and contributors from UC Berkeley's Human Compatible AI group
- **License**: See `LICENSE` file in this directory

## Description

Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the video game Overcooked. This code has been vendored (copied) into this project to provide the Overcooked environment for the RealtimeGym package.

## Notice

This code is **not maintained** as part of the RealtimeGym project. It is excluded from type checking and linting:
- `pyproject.toml`: Excluded from `[tool.ty.src]` and `[tool.ruff]`

For issues or contributions related to this code, please refer to the original repository: https://github.com/HumanCompatibleAI/overcooked_ai

## Citation

If you use the Overcooked environment in your research, please cite the original work.
