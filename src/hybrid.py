"""HYBRID (neuro-symbolic) pipeline:

    problem text --(LLM formalizer)--> IR --(validation)--> Z3 solver --(presentation)--> answer

This is the main thing your team builds. The plumbing is here; the parts marked TODO(team) are yours.
Suggested order of work:
    1. write FORMALIZER_PROMPT and get formalize() returning sensible IR for one simple problem;
    2. implement validate_ir();
    3. add ONE retry with feedback in solve();
    4. fix run_solver() so that "unique" and "multiple" can be told apart;
    5. (extension) replace the LLM-based present() with deterministic code for some problem types.

See docs/ir.md for the IR format.
"""
import ast
import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

import ir_solver
import llm
from answer import Answer


class HybridError(Exception):
    """The hybrid pipeline could not produce an answer (recorded by run.py as an error, not as status "none")."""


class IRModel(BaseModel):
    """JSON schema the formalizer must follow. Keep it in sync with docs/ir.md."""
    variables: dict[str, Any]
    all_different: list[list[str]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    objective: Optional[dict[str, str]] = None


# Compact IR description for the model. You may edit it, but keep it consistent with ir_solver.py.
IR_SPEC = """IR format (JSON):
{
  "variables": {"<name>": [lo, hi]  or  {"values": [v1, v2, ...]}},     integer variables and their domains
  "all_different": [["<name>", ...], ...],                                 groups whose values must all differ
  "constraints": ["<expression>", ...],                                    every expression must be true
  "objective": {"maximize": "<expression>"}  or  {"minimize": "<expression>"}   ONLY for optimisation problems
}
Expressions may use: integer numbers, variable names, + - * // %, == != < <= > >=, and, or, not, abs(), parentheses.
Inside arithmetic a comparison counts as 1 (true) or 0 (false), e.g. "(A == 1) + (B == 1) == 1".
Variable names must be identifiers: letters, digits and underscores only, no spaces."""
VAR_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SYSTEM = "You translate logic puzzles into a formal constraint representation. You answer only with JSON."

# TODO(team): this prompt is the heart of the hybrid pipeline. Things worth considering:
#   * how should the model choose variables? (one variable per entity? per attribute value?)
#   * how should it encode positions, days, "morning/afternoon", "next to", "between", "at most"?
#   * facts that the text does not state but that the model must know (calendar, working week, ...);
#   * a worked example (few-shot) on a puzzle that is NOT in your set;
#   * how to encode knights and knaves: a statement S by person X becomes  X == (S).
FORMALIZER_PROMPT = """Translate the puzzle below into the IR format. Do not solve the puzzle.
Important rules:
- Do NOT use Python lists [], tuples (), or quotes inside constraints. Use only raw variable names, integers, and operators.
- "all_different": Use ONLY when the puzzle explicitly requires values to be distinct. Otherwise, set "all_different": [].
- Boolean / Truth logic:
  * For truth values, binary roles, or conditions, use variables with domain [0, 1] (1 = true, 0 = false).
  * If a variable X asserts or is equivalent to a condition S, encode it strictly as: X == (S).
  * Examples:
    - If X says "Y is a knight": X == (Y == 1)
    - If X says "Y is a knave": X == (Y == 0)
    - If X says "at least one of Y, Z is a knave": X == (Y == 0 or Z == 0)
  * STRICT: Never use arithmetic operators (+, -, *) for logic. Use ONLY "or", "and", "not".

{ir_spec}

Example IR:
{{
 "variables": {{"Intercity": [1, 3], "Regional": [1, 3], "Express": [1, 3]}},
 "all_different": [["Intercity", "Regional", "Express"]],
 "constraints": ["Express != 1", "Regional < Intercity", "Intercity != 3"]
}}

Puzzle:
{text}
{feedback}"""


def formalize(problem: dict, feedback: tuple[str, str] | None = None) -> dict:
    """Ask the model for an IR encoding of the problem. `feedback` = error messages from a previous attempt."""
    fb = ""
    if feedback is not None:
        prev, why = feedback
        fb = (
            f"\n"
            f"Your previous encoding was:\n"
            f"{prev}\n"
            f"It was rejected because:\n"
            f"{why}\n"
            f"Fix these problems and write NEW IR.\n"
        )
    prompt = FORMALIZER_PROMPT.format(ir_spec=IR_SPEC, text=problem["text"], feedback=fb)
    parsed, raw = llm.ask_json(prompt, IRModel, system=SYSTEM)
    return parsed.model_dump(exclude_none=True)


def _validate_ir_variables(ir: dict, errors: list[str]) -> None:
    if "variables" not in ir:
        errors.append("IR must contain \"variables\" field")
        return

    for var_name, domain_or_values in ir["variables"].items():
        if not isinstance(var_name, str) or not VAR_NAME_PATTERN.fullmatch(var_name):
            errors.append(f"Invalid variable name: {var_name!r}")
            continue

        if isinstance(domain_or_values, list):
            if len(domain_or_values) != 2 \
                    or not isinstance(domain_or_values[0], int) \
                    or not isinstance(domain_or_values[1], int):
                errors.append(
                    f"Variable \"{var_name}\" has invalid domain: {domain_or_values}, "
                    f"must be an array of two numbers in format [low, high]"
                )
                continue
            lo, hi = domain_or_values
            if lo > hi:
                errors.append(f"Variable \"{var_name}\" has invalid domain: {domain_or_values}, lo must be less than hi")
        elif isinstance(domain_or_values, dict):
            if "values" not in domain_or_values \
                    or not all(isinstance(value, int) for value in domain_or_values["values"]):
                errors.append(
                    f"Variable \"{var_name}\" has invalid values: {domain_or_values}, "
                    f"must be an object in format {{\"values\": [v1, v2, ...]}}"
                )
        else:
            errors.append(f"Variable \"{var_name}\" has invalid domain or values: {domain_or_values}")


def _validata_ir_all_different(ir: dict, errors: list[str]) -> None:
    if "all_different" not in ir:
def _validate_ir_all_different(ir: dict, errors: list[str]) -> None:
        return

    all_diff = ir["all_different"]
    if not isinstance(all_diff, list):
        errors.append('"all_different" must be an array in format [["var1", "var2"], ...]')
        return

    valid_groups = []
    for var_names in all_diff:
        if not isinstance(var_names, list) or not all(isinstance(v, str) for v in var_names):
            errors.append('"all_different" must be a list in format [["var1", "var2"], ...]')
            continue

        has_unknown = False
        for var_name in var_names:
            if var_name not in ir["variables"]:
                errors.append(f'in "all_different": unknown variable "{var_name}"')
                has_unknown = True
        
        if has_unknown:
            continue

        possible_values = set()
        for var_name in var_names:
            dom = ir["variables"][var_name]
            if isinstance(dom, list) and len(dom) == 2:
                possible_values.update(range(dom[0], dom[1] + 1))
            elif isinstance(dom, dict) and "values" in dom:
                possible_values.update(dom["values"])

        if len(var_names) > len(possible_values):
            continue

        valid_groups.append(var_names)

    ir["all_different"] = valid_groups


def _validata_ir_objective(ir: dict, errors: list[str]) -> None:
def _validate_ir_objective(ir: dict, errors: list[str]) -> None:
    if "objective" not in ir:
        return

    obj = ir["objective"]
    if not isinstance(obj, dict) or not ("maximize" in obj or "minimize" in obj):
        errors.append(
            "\"objective\" must be an object "
            "in format {\"maximise\": \"<expression>\"} or {\"minimize\": \"<expression>\"}"
        )
        return

    expr = obj["maximize"] if "maximize" in obj else obj["minimize"]

    conv = ir_solver._Conv({var_name: 0 for var_name in ir["variables"]})

    try:
        conv(ast.parse(expr, mode="eval"))
    except (ValueError, SyntaxError) as e:
        errors.append(f"Error in objective: {e}")


def _validate_ir_constraints(ir: dict, errors: list[str]) -> None:
    if "constraints" not in ir:
        return

    constraints = ir["constraints"]
    if not isinstance(constraints, list) or not all(isinstance(c, str) for c in constraints):
        errors.append(f"\"constraints\" must be an array in format [\"expression\", ...]")
        return

    conv = ir_solver._Conv({var_name: 0 for var_name in ir["variables"]})

    for constraint in constraints:
        try:
            conv(ast.parse(constraint, mode="eval"))
        except (ValueError, SyntaxError) as e:
            errors.append(f"Error in constraint \"{constraint}\": {e}")


def validate_ir(ir: dict) -> list:
    """Return a list of human-readable error messages; an empty list means the IR is valid.
    Hint: ir_solver.build(ir) raises ValueError with a readable message for unknown variables and
    unsupported syntax - you can call it inside try/except and turn the exception into an error message.
    """

    errors = []

    _validate_ir_variables(ir, errors)
    _validate_ir_all_different(ir, errors)
    _validate_ir_objective(ir, errors)
    _validate_ir_constraints(ir, errors)

    try:
        ir_solver.build(ir)
    except (ValueError, SyntaxError) as e:
        err_msg = str(e)
        errors.append(f"Invalid expression or syntax: {err_msg}")

    return errors


def run_solver(ir: dict) -> dict:
    """Run Z3 on the IR. Returns {"status": ..., "solutions": [...], "objective": ...}.

    TODO(team): with limit=1 the solver stops after the first solution, so it reports "unique" even when
     several solutions exist. Think about what limit you need to tell "unique" from "multiple".
    """
    result = ir_solver.solve(ir, limit=2)
    return result


PRESENT_PROMPT = """A puzzle has been solved by a constraint solver. Rewrite the solver's result in the required
answer format. Do NOT solve the puzzle again - only translate the solver's assignment.

Puzzle:
{text}

Encoding used by the solver:
{ir}

Solver assignment (variable -> value):
{assignment}

{answer_format}"""


def present(problem: dict, ir: dict, result: dict) -> dict:
    """Turn the solver result into the required answer format.

    Provided baseline: the model translates the assignment. This is a weak point - the model can
    introduce errors AFTER the solver. Status and objective are always taken from the solver, never
    from the model. Extension: write deterministic code for problem types where the mapping is obvious.
    """
    status = result["status"]
    if status == "none":
        return Answer(status="none").model_dump()
    prompt = PRESENT_PROMPT.format(text=problem["text"], ir=json.dumps(ir, ensure_ascii=False),
                                   assignment=json.dumps(result["solutions"][0], ensure_ascii=False),
                                   answer_format=problem["answer_format"])
    parsed, raw = llm.ask_json(prompt, Answer)
    return Answer(status=status, solution=parsed.solution, objective=result.get("objective")).model_dump()


def solve(problem: dict) -> dict:
    """Returns {"answer": <Answer as dict>, "trace": {...}}. Raise HybridError if no answer can be produced."""
    trace = {"attempts": []}

    ir = formalize(problem)
    errors = validate_ir(ir)
    trace["attempts"].append({"ir": ir, "errors": errors})

    if errors:
        ir = formalize(problem, feedback=(json.dumps(ir), "\n".join(errors)))
        new_errors = validate_ir(ir)
        trace["attempts"].append({"ir": ir, "errors": new_errors})
        if new_errors:
            raise HybridError("invalid IR after retry: " + "; ".join(new_errors))

    try:
        result = run_solver(ir)
    except ValueError as e:                      # the solver rejected the IR
        raise HybridError(f"solver rejected the IR: {e}") from e
    trace["solver"] = {"status": result["status"], "n_solutions": len(result["solutions"]),
                       "objective": result.get("objective"),
                       "first_solution": result["solutions"][0] if result["solutions"] else None}

    answer = present(problem, ir, result)
    return {"answer": answer, "trace": trace}
