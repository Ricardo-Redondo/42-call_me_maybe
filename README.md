*This project has been created as part of the 42 curriculum by <login>.*

# call me maybe

## Description
Translates natural-language requests into structured function calls using
the small Qwen/Qwen3-0.6B model. For `"What is the sum of 2 and 3?"` the
program does not answer 5, it outputs:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers",
 "parameters": {"a": 2.0, "b": 3.0}}
```

The output is always valid JSON and always matches the function schema,
because generation is restricted token by token (constrained decoding)
instead of trusting the model to write JSON on its own.

## Instructions
Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```sh
make install      # uv sync (numpy, pydantic, local llm_sdk)
make run          # uv run python -m src
make lint         # flake8 + mypy
make moulinette   # prepare private set -> run program -> grade
make cpu-time     # timed run with the GPU hidden
```

Custom paths:
```sh
uv run python -m src --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

## Algorithm explanation
The model is only asked for the parts that are genuinely uncertain.
Everything else is written by Python.

1. **Prompt**: a short Qwen chat prompt with one line per function
   (`name(arg: type) - description`), the request, and an empty
   `<think></think>` block so the model answers directly. The output so
   far (`{"name": "`) is appended as if the model had written it.
2. **Vocabulary**: `vocab.json` is loaded via `get_path_to_vocab_file()`
   and every token is decoded from byte-level form (`Ġ` = space) to real
   text. From that, boolean numpy masks are built **once**: digits, `-`,
   `.`, number terminators, and for strings a per-token table saying if
   the token is valid inside a JSON string, if it closes it, and if it
   leaves a pending backslash.
3. **Function name**: the tokenized names form a trie. At each step, only
   the next tokens of names still possible are allowed (logits of every
   other token set to `-inf`). As soon as one name remains, the rest is
   filled in without calling the model.
4. **Arguments**: for each parameter, the key (`"a": `) is forced, then
   the value is generated with the mask of its type:
   - `number` / `integer`: grammar `-?digits(.digits)?` enforced by
     switching masks on/off; a `,`/`}` token ends it. `number` is always
     output as a float, `integer` as an int.
   - `string`: only tokens keeping a valid JSON string are allowed; after
     a `\` only valid escape characters are. The unescaped `"` ends the
     value, which is then decoded with `json.loads`.
   - `boolean`: choice between `true` and `false`.
5. The final JSON is built with pydantic + `json.dump`, so it is always
   parseable.

## Design decisions
- **Python writes the structure, the model writes the values.** A forced
  token needs no forward pass, so this is both the strictest constraint
  and the biggest speed-up.
- **Masks precomputed with numpy**: a step is one `np.where` + `argmax`
  over 151,936 logits, no Python loop over the vocabulary.
- **Logit cache** keyed on the token sequence (same idea as `lru_cache`).
- **Validation before model loading**: bad input files fail instantly.
- **One output entry per prompt**, even if a prompt fails, so results stay
  aligned with the inputs.
- Only public `llm_sdk` methods are used; the SDK is unmodified.

## Performance analysis
| | Public set | Private set |
|---|---|---|
| Moulinette score | 10/11 (90.9%) | 11/11 (100%) |
| JSON validity | 100% | 100% |

Speed (11 prompts): ~22 s with a GPU, ~105 s CPU-only (about 1 s per model
call because the SDK has no KV cache), well under the 5-minute limit.
Decoding is greedy (argmax), so runs are deterministic.

The remaining public miss: "replace vowels with asterisks" gives the word
`asterisk` as replacement instead of `*`.

## Challenges faced
- **No KV cache in the SDK**: each call re-reads the whole prompt, so
  speed comes from calling the model as rarely as possible and keeping
  the prompt short.
- **Vocab size vs logits size**: `vocab.json` has 151,643 entries but the
  logits vector 151,936 (special tokens); masks are padded to the logits
  size.
- **Types**: the grader checks `isinstance(x, float)` for `number`, so
  `2` must be written `2.0`.
- **Quotes and backslashes in values** (`Say "hello"`,
  `C:\Users\...`): solved by enforcing valid JSON escapes in the mask.
- **Leading space**: the model tends to write `" /home"` after the
  opening quote; the leading space is stripped.

## Testing strategy
- The provided moulinette on the public and private sets (`make moulinette`).
- Error cases by hand: missing file, invalid JSON, wrong structure, empty
  function list, unwritable output path. Each gives a one-line error and
  exit code 1, never a traceback.
- `flake8` and `mypy --strict`.

## Example usage
```sh
$ uv run python -m src
model ready in 16.1s
[1/11]  1.18s   7 calls  fn_add_numbers {"a": 2.0, "b": 3.0}
[3/11]  0.28s   4 calls  fn_greet {"name": "shrek"}
...
done in 22.0s -> data/output/function_calling_results.json

$ uv run python -m src --input missing.json
error: cannot read missing.json: No such file or directory
```

## Resources
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Byte-level BPE (GPT-2 tokenizer)](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- [JSON specification](https://www.json.org/)
- [Outlines: efficient guided generation (paper)](https://arxiv.org/abs/2307.09702)
- [Pydantic docs](https://docs.pydantic.dev/)

**AI usage**: TODO: describe honestly which parts you used AI for.
