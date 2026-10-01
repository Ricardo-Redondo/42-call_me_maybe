"""Entry point: uv run python -m src [--functions_definition ...] ..."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from src.models import FunctionCall, FunctionDef, PromptEntry


class InputError(Exception):
    """Input file missing or invalid (message is user-facing)."""


def fmt_time(seconds: float) -> str:
    """Format a duration as minutes:seconds, e.g. 105.3 -> "1:45.30".

    divmod(105.3, 60) gives (1.0, 45.3): whole minutes and the rest.
    {s:05.2f} = 2 decimals, padded with zeros to 5 characters, so
    5.3 seconds shows as "05.30" (like a clock).
    """
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:05.2f}"


def parse_args() -> argparse.Namespace:
    """CLI flags with the subject's default paths.

    argparse reads the command line: `--input foo.json` ends up in
    args.input; if the flag is not given, the default is used.
    """
    p = argparse.ArgumentParser(description="call me maybe")
    p.add_argument("--functions_definition",
                   default="data/input/functions_definition.json")
    p.add_argument("--input",
                   default="data/input/function_calling_tests.json")
    p.add_argument("--output",
                   default="data/output/function_calling_results.json")
    return p.parse_args()


def load_list(path: str, adapter: TypeAdapter[Any]) -> Any:
    """Read `path` and validate it with `adapter`.

    `adapter` is e.g. TypeAdapter(list[FunctionDef]): pydantic parses
    the JSON text AND checks that it is a list of FunctionDef in one go.
    Every problem is turned into an InputError with a readable message.
    """
    try:
        # `with` closes the file automatically, even on an error
        with open(path, encoding="utf-8") as f:
            return adapter.validate_json(f.read())
    except OSError as e:
        # file missing, no permission, path is a directory...
        # e.strerror is the readable reason ("No such file...").
        # `raise X from e` keeps the original error attached (useful
        # when debugging) while showing our clearer message.
        raise InputError(f"cannot read {path}: {e.strerror}") from e
    except ValidationError as e:
        # broken JSON, or valid JSON with the wrong shape.
        # e.errors() is a list of problems; we report the first one.
        # err["loc"] says where, e.g. (0, "prompt") = first item's prompt
        err = e.errors()[0]
        # (0, "prompt") -> "0/prompt"; empty -> "file" (`or` gives the
        # right side when the left side is empty)
        where = "/".join(str(x) for x in err["loc"]) or "file"
        raise InputError(f"invalid content in {path} "
                         f"(at {where}): {err['msg']}") from e


def main() -> int:
    """Load inputs, run the decoder on each prompt, write the output.

    Returns the exit code: 0 on success, 1 on any error.
    """
    start = time.time()
    args = parse_args()
    # 1. Read and validate the inputs BEFORE loading the model, so a
    #    bad file fails in 0.1 s instead of after the model loads.
    try:
        functions: list[FunctionDef] = load_list(
            args.functions_definition, TypeAdapter(list[FunctionDef]))
        prompts: list[PromptEntry] = load_list(
            args.input, TypeAdapter(list[PromptEntry]))
    except InputError as e:
        # file=sys.stderr: errors go to the error stream, not stdout
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not functions:
        # an empty list counts as False
        print("error: no functions defined", file=sys.stderr)
        return 1

    # 2. Load the model and build the vocabulary masks.
    try:
        # imported here so the input checks above don't wait for
        # torch/transformers to load (they are slow to import)
        from src.decoder import Decoder
        from src.llm import LLM
        from src.vocab import Vocab
        llm = LLM()
        # one throwaway call just to learn the logits size (151936)
        n_logits = len(llm.logits(llm.encode("a")))
        decoder = Decoder(llm, Vocab(llm.vocab_path(), n_logits), functions)
    except Exception as e:
        # no internet to download the model, out of memory...
        print(f"error: could not load the model: {e}", file=sys.stderr)
        return 1
    print(f"model ready in {fmt_time(time.time() - start)}",
          file=sys.stderr)

    # 3. One FunctionCall per prompt, in the same order as the input
    #    (the moulinette compares answer i with test i).
    calls: list[FunctionCall] = []
    # enumerate(..., 1) counts from 1 instead of 0, for "[1/11]"
    for i, entry in enumerate(prompts, 1):
        t = time.time()
        before = llm.calls
        try:
            call = decoder.run(entry.prompt)
        except Exception as e:
            # never crash on one prompt: write a fallback entry instead
            # so the output stays aligned with the input
            print(f"  warning: {e}", file=sys.stderr)
            call = FunctionCall(prompt=entry.prompt,
                                name=functions[0].name, parameters={})
        calls.append(call)
        # progress line: time, number of model runs, result.
        # {n:3d} = whole number padded to 3 characters (aligns columns)
        print(f"[{i}/{len(prompts)}] {fmt_time(time.time() - t)} "
              f"{llm.calls - before:3d} calls  {call.name} "
              f"{json.dumps(call.parameters)}", file=sys.stderr)

    # 4. Write the output file (creating data/output/ if needed).
    out = Path(args.output)
    try:
        # out.parent = the folder part of the path (data/output);
        # parents=True creates missing folders on the way,
        # exist_ok=True doesn't complain if it already exists
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:     # "w" = write
            # model_dump() turns each pydantic object back into a dict;
            # [... for c in calls] does it for every call ("list
            # comprehension"); indent=2 makes the file readable
            json.dump([c.model_dump() for c in calls], f, indent=2)
    except OSError as e:
        print(f"error: cannot write {out}: {e.strerror}", file=sys.stderr)
        return 1
    print(f"done in {fmt_time(time.time() - start)} -> {out}",
          file=sys.stderr)
    return 0


# True only when this file is run directly (python -m src),
# not when it is imported by another file
if __name__ == "__main__":
    try:
        # sys.exit(code) ends the program with that exit code
        # (0 = success, anything else = error; `echo $?` shows it)
        sys.exit(main())
    except KeyboardInterrupt:
        # Ctrl+C: clean message instead of a traceback
        # (130 is the usual exit code for "stopped by Ctrl+C")
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
