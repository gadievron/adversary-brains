"""Command-line interface: `python -m brainkit <command>`.

The shipped CLI is `validate` (the appendix-1 CI contract) — that is the whole
command surface, by decision (CAP-cli-validate-only). The rest of the engine
(`screen`, `match`, `assess`, `weights`, `extract`, `discover`, `loop`) is a
library exercised by the experiment scripts and tests, NOT by CLI subcommands;
argparse rejects an unknown subcommand rather than pretending one exists.
"""
from __future__ import annotations

import argparse
import sys

from .schema import DEFAULT_SCHEMA_PATH, validate_repo


def _cmd_validate(args: argparse.Namespace) -> int:
    errors = validate_repo(args.root, args.schema)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"FAIL: {len(errors)} error(s)", file=sys.stderr)
        return 1
    print(f"OK: all canonical brains under {args.root} validate "
          "(study/ prose artifacts excluded; schema, LOCAL-field rejection, "
          "id uniqueness, related-id resolution, entity refs)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainkit", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a brains/ tree (CI contract)")
    validate.add_argument("root", help="path to the brains/ tree")
    validate.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH),
                          help="path to brain-0.1.schema.json")
    validate.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
