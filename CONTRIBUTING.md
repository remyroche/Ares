# Contributing

## Development setup

Install the locked development environment and enable the repository hooks once per clone:

```bash
poetry install --with dev
poetry run pre-commit install
```

Run all local checks before a broad refactor or when updating the hook configuration:

```bash
poetry run pre-commit run --all-files
```

Run the smallest relevant test target while developing. For example:

```bash
poetry run pytest tests/test_validate_commit_message.py
```

## Commit messages

Use Conventional Commits with a concise subject of 72 characters or fewer and no
trailing period:

```text
feat(inference): add latency guard
fix(policy): handle empty candidate set
test: cover commit message validation
```

The commit-msg hook permits Git-generated merge, revert, fixup, and squash
subjects.
