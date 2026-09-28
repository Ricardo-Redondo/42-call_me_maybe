"""Constrained decoding: pick the function, then generate each argument.

Python builds the JSON. The model only produces the parts that are
actually uncertain: which function, and each value. Everything forced
(braces, keys, quotes, the tail of a function name) is appended as
tokens WITHOUT calling the model. That's constrained decoding at its
most extreme: when only one token is valid, skip the forward pass.

Words used below:
    ids     list of token ids = the text the model "sees" so far
    logits  one score per token id, higher = model prefers it next
    mask    True/False per token id, False = forbidden (score -> -inf)
"""

import json
from typing import Any

import numpy as np
import numpy.typing as npt

from src.llm import LLM
from src.models import FunctionCall, FunctionDef
from src.vocab import Vocab

# Safety limit: a value can never be longer than this many tokens,
# so a confused model can't loop forever.
MAX_VALUE_TOKENS = 48

# Qwen3 chat format. <|im_start|>/<|im_end|> are special tokens that
# mark who is talking (system / user / assistant).
# {functions} is a placeholder, filled in later with .format().
# "\\\\d+" in Python source = the text \\d+ in the prompt, which is
# how the regex \d+ looks inside JSON (backslashes are doubled).
SYSTEM = (
    "<|im_start|>system\n"
    "Pick the function matching the request and extract its arguments "
    "exactly from the request. Answer in JSON. Regex arguments are "
    "Python regular expressions (e.g. \\\\d+, [abc]); replacements are "
    "the literal characters to insert, never their names.\n"
    "Functions:\n{functions}<|im_end|>\n"
    "<|im_start|>user\n"
)
# The empty <think></think> tells Qwen3 to skip its "reasoning" text
# and answer directly (saves a lot of tokens = time).
USER = (
    "{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


class Decoder:
    """Turn one natural-language prompt into a FunctionCall."""

    def __init__(self, llm: LLM, vocab: Vocab,
                 functions: list[FunctionDef]) -> None:
        """Store deps and pre-tokenize everything that never changes."""
        self.llm = llm
        self.vocab = vocab
        # name -> definition, to find the parameters of the chosen one.
        self.functions = {f.name: f for f in functions}
        # name -> its token ids, e.g. "fn_greet" -> [id, id, id]
        self.name_ids = {f.name: llm.encode(f.name) for f in functions}
        # One short line per function, e.g.
        #   fn_add_numbers(a: number, b: number) - Add two numbers...
        lines = []
        for f in functions:
            # f"{k}: {v.type}" builds "a: number" for each parameter,
            # ", ".join(...) glues them: "a: number, b: number"
            args = ", ".join(f"{k}: {v.type}" for k, v in f.parameters.items())
            lines.append(f"{f.name}({args}) - {f.description}")

        # The system part is the same for every prompt: encode it once.
        # "\n".join(lines) puts one function per line.
        self.header_ids = llm.encode(SYSTEM.format(functions="\n".join(lines)))
        # Token ids we need to refer to directly. encode returns a
        # list, [0] takes its first (here: only) id.
        self.quote_id = llm.encode('"')[0]
        self.true_id = llm.encode("true")[0]
        self.false_id = llm.encode("false")[0]

    def masked_argmax(self, ids: list[int],
                      mask: npt.NDArray[np.bool_]) -> int:
        """Most likely token among those allowed by `mask`.

        This is THE constrained-decoding step: run the model, set every
        forbidden token's score to -infinity, take the best remaining.
        """
        logits = self.llm.logits(ids)
        # np.where(mask, a, b): for each position, take a if mask is
        #   True there, else b. So allowed tokens keep their score and
        #   forbidden ones get -infinity (can never be the maximum).
        # .argmax() returns the POSITION of the biggest value, which
        #   is exactly the token id. int(...) turns numpy's number
        #   type into a normal Python int.
        return int(np.where(mask, logits, -np.inf).argmax())

    def only(self, *token_ids: int) -> npt.NDArray[np.bool_]:
        """Mask allowing exactly `token_ids`.

        The `*` means "any number of arguments": only(5, 9, 12) gives
        token_ids == (5, 9, 12).
        """
        mask = np.zeros(len(self.vocab.texts), dtype=bool)  # all False
        # numpy lets you set many positions at once:
        # mask[[5, 9, 12]] = True sets those three slots to True
        mask[list(token_ids)] = True
        return mask

    def choose_function(self, ids: list[int]) -> str:
        """Trie walk over the tokenized function names.

        Only calls the model while more than one name is still possible;
        a closing quote is the "end" choice when a name is a prefix of
        another one.

        Example with fn_add_numbers / fn_greet / fn_get_square_root:
            step 0: all start with "fn"   -> forced, no model call
            step 1: "_add" / "_g"...      -> model chooses among them
            once one name is left         -> stop, no more calls
        """
        picked: list[int] = []          # tokens of the name chosen so far
        candidates = list(self.name_ids)    # list(dict) = its keys
        while len(candidates) > 1:
            step = len(picked)
            # Group the remaining names by their next token:
            #   {next_token_id: [names that continue with it]}
            nxt: dict[int, list[str]] = {}
            for name in candidates:
                toks = self.name_ids[name]
                # if this name is already complete, its "next token"
                # is the closing quote of "name": "..."
                # (x if condition else y  is a one-line if/else)
                key = toks[step] if step < len(toks) else self.quote_id
                # setdefault: if `key` isn't in the dict yet, create it
                # with an empty list; then append the name to that list
                nxt.setdefault(key, []).append(name)
            if len(nxt) == 1:
                # every candidate continues the same way: forced token,
                # no need to ask the model.
                # next(iter(nxt)) = "the first (and only) key of nxt"
                tok = next(iter(nxt))
            else:
                # real choice: the model picks, but only among these.
                # ids + picked = the prompt followed by the name so far;
                # *nxt passes each key of nxt as a separate argument
                tok = self.masked_argmax(ids + picked, self.only(*nxt))
            candidates = nxt[tok]   # keep only names using that token
            if tok == self.quote_id:
                break
            picked.append(tok)
        return candidates[0]

    def gen_number(self, ids: list[int], integer: bool) -> float | int:
        """Generate a number: float for "number", int for "integer".

        Grammar enforced by the masks: -? digits (. digits)?
        The terminator (',' or '}' token) is only a stop signal.

        The moulinette checks isinstance(x, float) for "number", so we
        must return 2.0 and not 2.
        """
        v = self.vocab      # short alias, just to type less
        text = ""           # the number written so far, e.g. "-12."
        ids = list(ids)     # copy: we append to it locally
        for _ in range(MAX_VALUE_TOKENS):
            # any(...) is True if at least one character is a digit
            has_digit = any(c.isdigit() for c in text)
            # Build the mask for this step. Digits are always allowed.
            # .copy() is needed: without it we would modify the shared
            # digit_mask itself and break it for the next numbers.
            mask = v.digit_mask.copy()
            # `a |= b` is "a = a OR b" position by position: it adds
            # the True slots of b to a (= allows those tokens too).
            if not text:
                # "-" only as the very first character
                mask |= v.minus_mask
            if has_digit and not integer and "." not in text:
                # one "." max, after at least one digit, never in ints
                mask |= v.dot_mask
            if has_digit and not text.endswith("."):
                # may stop only once the number is complete
                # ("", "-" and "12." are not valid numbers)
                mask |= v.end_number_mask
            tok = self.masked_argmax(ids, mask)
            if v.end_number_mask[tok]:
                break           # model says the number is finished
            text += v.texts[tok]
            ids.append(tok)
        # rstrip(".") removes "." characters from the END only.
        text = text.rstrip(".")  # in case we hit MAX_VALUE_TOKENS on "12."
        try:
            # float("265") gives 265.0, int("7") gives 7
            return int(text) if integer else float(text)
        except ValueError:
            # only possible if the limit was hit before any digit
            return 0 if integer else 0.0

    def gen_string(self, ids: list[int]) -> str:
        """Generate a JSON string body until the model closes the quote.

        Masks only allow valid JSON escapes, so json.loads on the body
        always succeeds.

        Example: for C:\\Users the model must write C:\\\\Users (JSON
        escaped); json.loads turns it back into C:\\Users.
        """
        v = self.vocab
        body = ""       # raw JSON text between the quotes
        state = 0       # 0 = normal, 1 = just wrote a lone '\'
        ids = list(ids)
        for _ in range(MAX_VALUE_TOKENS):
            # allowed tokens depend on whether we are mid-escape:
            # v.str_valid[0] is the normal mask, [1] the after-'\' one
            tok = self.masked_argmax(ids, v.str_valid[state])
            close = int(v.str_close[state][tok])
            if close >= 0:
                # token contains the closing quote (e.g. 'lo",'):
                # keep the part before it and stop.
                # text[:close] = characters from the start up to (not
                # including) position `close`: 'lo",'[:2] == 'lo'
                body += v.texts[tok][:close]
                break
            body += v.texts[tok]
            # True/False -> 1/0, the new state for the next token
            state = int(v.str_escaped[state][tok])
            ids.append(tok)
        if state:
            # limit hit right after a '\': drop it to keep JSON valid.
            # body[:-1] = everything except the last character
            body = body[:-1]
        # the model often "types" a space after the opening quote;
        # lstrip(" ") removes spaces from the START only
        body = body.lstrip(" ")
        try:
            # turn JSON escapes (\" \\ \n ...) into real characters:
            # we put the quotes back around the body so it is a full
            # JSON string, then let json.loads decode it
            value: str = json.loads(f'"{body}"')
            return value
        except ValueError:
            return body

    def gen_bool(self, ids: list[int]) -> bool:
        """Pick between the first token of 'true' and of 'false'."""
        tok = self.masked_argmax(ids, self.only(self.true_id, self.false_id))
        # a comparison already gives True/False
        return tok == self.true_id

    def run(self, prompt: str) -> FunctionCall:
        """Full pipeline for one prompt.

        `out` is the JSON answer written so far. We write the fixed
        parts ourselves and let the model fill the blanks:
            {"name": "<model>", "parameters": {"a": <model>, ...
        """
        # list + list = one longer list (system part, then this prompt)
        base = self.header_ids + self.llm.encode(USER.format(prompt=prompt))
        # 1. function name
        out = '{"name": "'
        name = self.choose_function(base + self.llm.encode(out))
        out += name + '", "parameters": {'
        # 2. each parameter, in the order of the definition
        params: dict[str, Any] = {}     # Any = "a value of any type"
        # enumerate gives (0, first_item), (1, second_item)...
        # each item is itself a (key, spec) pair from .items()
        for i, (key, spec) in enumerate(
                self.functions[name].parameters.items()):
            # forced text: ', "key": ' (no comma before the first one).
            # json.dumps("a") gives '"a"' (the name with its quotes)
            # i is 0 for the first parameter, and 0 counts as False
            out += (", " if i else "") + json.dumps(key) + ": "
            if spec.type == "string":
                out += '"'      # opening quote is forced too
            # the model sees the prompt + everything written so far
            ids = base + self.llm.encode(out)
            value: Any
            if spec.type == "string":
                value = self.gen_string(ids)
            elif spec.type in ("number", "integer"):
                # the 2nd argument is True only for "integer"
                value = self.gen_number(ids, spec.type == "integer")
            else:
                value = self.gen_bool(ids)
            params[key] = value
            # write the value into `out` properly JSON-formatted, so the
            # next parameter sees a clean context (drop our opening '"',
            # json.dumps adds both quotes)
            out = out.rstrip('"') + json.dumps(value)
        return FunctionCall(prompt=prompt, name=name, parameters=params)
