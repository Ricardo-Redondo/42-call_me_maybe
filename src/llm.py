"""Thin wrapper around llm_sdk: cached logits as numpy arrays."""

import numpy as np
import numpy.typing as npt

from llm_sdk import Small_LLM_Model


class LLM:
    """Wraps Small_LLM_Model using only its public methods.

    The SDK has no KV cache: every call re-reads the WHOLE sequence
    (~0.5-1.5 s per call on CPU, ~0.08 s on GPU). So:
      * keep the prompt short,
      * never call the model when only one token is valid,
      * never compute the same sequence twice (cache below).
    """

    def __init__(self) -> None:
        """Load the model (Qwen/Qwen3-0.6B, the SDK default)."""
        self.model = Small_LLM_Model()

        # token sequence -> logits already computed for it.
        # npt.NDArray[np.float32] is just the type hint for
        # "a numpy array of 32-bit floats".
        self._cache: dict[tuple[int, ...], npt.NDArray[np.float32]] = {}

        # how many real model runs we did (printed per prompt in main)
        self.calls = 0

    def encode(self, text: str) -> list[int]:
        """Text -> list of token ids."""
        # the SDK returns a 2D tensor of shape [1, n] (a "batch" of one
        # sentence): take row 0 and turn it into a plain Python list
        ids: list[int] = self.model.encode(text)[0].tolist()
        return ids

    def logits(self, ids: list[int]) -> npt.NDArray[np.float32]:
        """Next-token logits for `ids`, memoized on tuple(ids).

        Same idea as functools.lru_cache, but a dict on the instance
        (lru_cache on a method keeps `self` alive and needs hashable args).
        """
        # lists can't be dict keys (they can change), tuples can
        key = tuple(ids)
        # .get(key) returns the stored value, or None if not there
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self.calls += 1
        # the SDK gives a Python list of ~151k floats; a numpy array
        # lets the decoder mask and argmax it in one fast operation
        out = np.array(self.model.get_logits_from_input_ids(ids),
                       dtype=np.float32)
        self._cache[key] = out
        return out

    def vocab_path(self) -> str:
        """Path to vocab.json (token string -> id)."""
        # str(...) only reassures mypy that the SDK returns a string
        return str(self.model.get_path_to_vocab_file())
