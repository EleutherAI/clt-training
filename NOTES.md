# INSTALLATION HELP

This section details any issue with setting up the environment.

- Python Versions Tested:
  - Python 3.13.0
- **FOR ZSH USERS:** `pip install -e .[dev]` (in original README), no matches found error.
  - Run `pip install -e .'[dev]'` instead

# MODIFICATIONS

This section details the changes made to the original repo to make things work.

- Changed dependency of bitsandbytes to `bitsandbytes>=0.42.0` in `pyproject.toml`.
