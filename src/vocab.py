"""Token id -> text table and precomputed numpy masks.

The model does not know about letters, it only knows ~151k "tokens"
(pieces of text, each with an id). To constrain generation we need to
know, for every token id, what text it stands for, and then which
tokens are allowed in each situation (inside a number, inside a
string...). A "mask" is a numpy array of True/False with one slot per
token id: True = allowed.

Every mask is built ONCE at startup. Per step we then only do
`np.where(mask, logits, -np.inf).argmax()`, never a Python loop
over the ~151k tokens.
"""

import json

import numpy as np

# Characters that may follow a backslash inside a JSON string
# (\" \\ \/ \b \f \n \r \t). \u is left out on purpose: it would
# need 4 hex digits after it and we don't need it for our inputs.
# Note: in Python source '\\' is ONE backslash, so this string is
# the 8 characters  "  \  /  b  f  n  r  t
JSON_ESCAPES = '"\\/bfnrt'


def byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2 bytes_to_unicode ('Ġ' -> space, 'Ċ' -> newline).

    vocab.json keys are byte-level strings: printable bytes map to
    themselves, the other bytes were shifted to chr(256 + n).

    Example: the token for " hello" is stored as "Ġhello" in vocab.json,
    because a space (byte 32) is not printable, so it was shifted to
    chr(256 + 32) = 'Ġ'. This table undoes that shift.
    """
    # Bytes that were kept as themselves: the printable ASCII range
    # '!'..'~' and two printable Latin-1 ranges.
    # ord("!") gives the number of a character (33).
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))

    # printable char -> its own byte value.
    # This is a "dict comprehension": {key: value for item in list}.
    # chr(33) is the reverse of ord: number -> character ("!").
    table = {chr(b): b for b in bs}

    # Every other byte (space, newline, control chars...) was given a
    # replacement character chr(256), chr(257), ... in increasing order.
    n = 0
    for b in range(256):
        if b not in bs:
            table[chr(256 + n)] = b
            n += 1
    return table


def scan_string_token(text: str, escaped: bool) -> tuple[bool, int, bool]:
    """Simulate appending `text` inside a JSON string.

    Used at startup to pre-classify every token for string generation.

    Args:
        text: decoded token text.
        escaped: True if the previous char was an unfinished backslash.

    Returns:
        (valid, close_index, escaped_after). close_index is the index of
        the unescaped '"' that ends the string, or -1 if still open.

    Examples (escaped=False):
        'hello'  -> (True, -1, False)   plain text, string stays open
        'lo",'   -> (True, 2, False)    the '"' at index 2 closes it
        'a\\'    -> (True, -1, True)    ends with a lone backslash
        '\\x'    -> (False, -1, False)  \\x is not a valid JSON escape
    """
    # enumerate gives (position, character) pairs: (0,'l'), (1,'o')...
    for i, ch in enumerate(text):
        if escaped:
            # previous char was '\': this char must be a legal escape
            if ch not in JSON_ESCAPES:
                return False, -1, False
            escaped = False
        elif ch == "\\":
            # start of an escape sequence, the next char decides
            escaped = True
        elif ch == '"':
            # an unescaped quote = end of the string value
            return True, i, False
        # ord(ch) < 0x20: character codes 0-31 are invisible control
        # characters (0x20 is hexadecimal for 32, the space)
        elif ord(ch) < 0x20 or ch == "\ufffd":
            # raw control chars (newline, tab...) are illegal in JSON
            # strings; '\ufffd' or '�' means "broken utf-8 piece"
            return False, -1, False
    # got to the end of the token without closing the string
    return True, -1, escaped


class Vocab:
    """Decoded token texts plus the boolean masks the decoder needs."""

    def __init__(self, vocab_path: str, n_logits: int) -> None:
        """Load vocab.json and build masks.

        vocab.json has fewer entries (151643) than the logits vector
        (151936): every mask is sized `n_logits` and padded with False.
        """
        # texts[id] = real text of token `id` ("" for special tokens,
        # which are not in vocab.json and so are never allowed).
        # [""] * n makes a list of n empty strings.
        self.texts: list[str] = [""] * n_logits
        decoder = byte_decoder()
        with open(vocab_path, encoding="utf-8") as f:
            # vocab.json is one big {"token_string": id, ...} object
            raw: dict[str, int] = json.load(f)
        for tok, idx in raw.items():
            if idx < n_logits:
                # 'Ġhello' -> b' hello' -> ' hello'
                # For each character c of the token, look up its byte
                # (decoder.get(c, 0) = table value, or 0 if missing),
                # and bytes(...) packs those numbers into raw bytes.
                data = bytes(decoder.get(c, 0) for c in tok)
                # a token can be half of a multi-byte char (emoji...),
                # errors="replace" turns that into '�' instead of
                # crashing
                self.texts[idx] = data.decode("utf-8", errors="replace")

        size = n_logits
        # np.zeros(size, dtype=bool) = an array of `size` False values.
        # We then flip to True the ids that belong to each group.
        # - number masks
        self.digit_mask = np.zeros(size, dtype=bool)       # "0".."9"...
        self.minus_mask = np.zeros(size, dtype=bool)       # "-"
        self.dot_mask = np.zeros(size, dtype=bool)         # "."
        self.end_number_mask = np.zeros(size, dtype=bool)  # ",", "}"...

        # - string tables
        # Indexed by state: [0] = normal, [1] = right after a '\'.
        # str_valid[s][id]   -> may token `id` be used in state s?
        # str_close[s][id]   -> index of the closing '"' in it, or -1
        # str_escaped[s][id] -> does it end with a pending '\'?
        # [... for _ in range(2)] builds a list of 2 separate arrays.
        self.str_valid = [np.zeros(size, dtype=bool) for _ in range(2)]

        # np.full(size, -1) = an array of `size` values, all -1
        self.str_close = [np.full(size, -1, dtype=np.int32)
                          for _ in range(2)]
        self.str_escaped = [np.zeros(size, dtype=bool) for _ in range(2)]

        # One pass over the whole vocabulary, done once at startup.
        for idx, text in enumerate(self.texts):
            if not text:
                # empty string = special token: leave it all False
                continue

            # .isascii() rules out digits of other alphabets,
            # .isdigit() = only digit characters
            if text.isascii() and text.isdigit():
                self.digit_mask[idx] = True
            elif text == "-":
                self.minus_mask[idx] = True
            elif text == ".":
                self.dot_mask[idx] = True

            # text[0] in ",}" = first character is ',' or '}'
            # text.strip() = text without spaces around it
            elif text[0] in ",}" or text.strip() in (",", "}"):
                # what the model would write after a number in JSON;
                # we use it as the "number is finished" signal
                self.end_number_mask[idx] = True

            for state in (0, 1):
                # bool(0) is False, bool(1) is True
                ok, close, esc = scan_string_token(text, bool(state))
                self.str_valid[state][idx] = ok
                self.str_close[state][idx] = close
                self.str_escaped[state][idx] = esc
