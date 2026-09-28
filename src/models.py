"""Pydantic models for input files and output entries.

Pydantic checks the data when an object is created: if a field is
missing or has the wrong type, it raises ValidationError instead of
letting bad data travel through the program.
"""

from typing import Any, Literal

from pydantic import BaseModel

# The only parameter types we know how to generate. Anything else in
# functions_definition.json (e.g. "array") is rejected at load time.
# Literal[...] means "must be exactly one of these values".
ParamType = Literal["number", "integer", "string", "boolean"]


# Inheriting from BaseModel is what makes a class a pydantic model:
# each `name: type` line below becomes a checked field.
class ParamSpec(BaseModel):
    """Type of one parameter (or return value) of a function."""

    type: ParamType


class FunctionDef(BaseModel):
    """One entry of functions_definition.json.

    Example:
        {"name": "fn_greet", "description": "Generate a greeting...",
         "parameters": {"name": {"type": "string"}},
         "returns": {"type": "string"}}
    """

    name: str
    description: str
    # dict[str, ParamSpec]: keys are parameter names, each value is
    # checked as a ParamSpec (so {"type": "number"})
    parameters: dict[str, ParamSpec]
    returns: ParamSpec


class PromptEntry(BaseModel):
    """One entry of function_calling_tests.json: {"prompt": "..."}."""

    prompt: str


class FunctionCall(BaseModel):
    """One entry of the output file: exactly prompt, name, parameters."""

    prompt: str
    name: str
    # Any: values can be float, int, str or bool depending on the
    # function, so we don't restrict the type here
    parameters: dict[str, Any]
