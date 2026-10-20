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
from copy import deepcopy
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

RETRY_TEMPERATURE = 0.4
STATUS_RANK = {"none": 0, "multiple": 1, "unique": 2, "optimal": 2}

FORMALIZER_PROMPT = """Translate the puzzle below into the IR JSON format. Do not solve it: encode
what it says, then stop. Return one JSON object and nothing else.

Variables and domains
- One variable per entity (or per attribute), named after it. Names use ASCII letters, digits and
  underscores only and must not start with a digit.
- An integer domain is [low, high] inclusive; a fixed value is [n, n]; a yes/no variable is [0, 1]
  with 1 = true; a non-contiguous set uses {{"values": [v1, v2]}}.
- For an ordered set (chairs, hours, days, ranks) use the numbers the puzzle itself gives; for a
  working week use 1 = Monday .. 5 = Friday, and invent a longer cycle only if the puzzle names one.

all_different
- Only for "one of each" puzzles, where every value is used exactly once: put the whole set of
  those variables into ONE group.
- Never for yes/no roles or independently selectable features - several people can be knaves and
  several features can be built, so those variables must be free to share values.
- Every name in a group must be a key of "variables".

Each constraint is one Boolean condition, written with ==, !=, <, <=, >, >=, and, or, not and
parentheses (never &, |, >> or arrows). Encode each clause exactly once, as written:
- One clause in, one constraint out, in the form that clause's own wording calls for. Never glue
  two clauses into one constraint, and never wrap a clause in "... <= 1" unless that clause itself
  says the two named things cannot BOTH hold: "<= 1" only ever means "not both", so fusing it onto
  another clause silently forbids that pair and forces the rest of the puzzle to fit around it.
- "A implies B", "if A then B", "A only if B", "A can only be built if B is built"
    -> "A <= B" for yes/no variables, equivalently not A or B.
    Never "A and B", and never "A + B >= 1": that says "at least one", which is weaker.
- "A and B cannot both ..." -> "(A == 1) + (B == 1) <= 1", never >= 1.
- "at least k of these conditions" -> "(A == 1) + (B == 1) + ... >= k", every comparison
  parenthesised. "at most k" and "exactly k" use <= k and == k the same way.
- A quoted statement, "X says: S", is a statement BY X and becomes ONE constraint "X == (S)",
  with S written as a Boolean condition using the rules above. Do not restate S as a fact of its
  own, and do not chain the speakers' statements into implications between their roles. Such roles
  are free variables: never all_different over them and never a cap on how many hold one of them -
  the puzzle never says how many of each kind there are.
- A clause that names ONE member of a sequence by description - "the first", "the middle",
  "the last", "the one just before the next boundary" - is ONE equality on that one value, e.g.
  "the middle of the week" -> "Mid == 3" for a five-day week. Not a list of exclusions.
- Ordering and exclusions are ordinary comparisons. Around a round table with n numbered seats:
  "next to" -> "(abs(A - B) == 1) or (abs(A - B) == n-1)";
  "directly opposite" (n even) -> "abs(A - B) == n/2";
  "immediately clockwise from B" -> "(A == B + 1) or (A == B - (n-1))", counter-clockwise with
  the two signs swapped.
- A budget, a cost limit or a cap on how many items may be chosen is a CONSTRAINT, e.g.
  "5*Lamp + 4*Mug <= 12" or "Lamp + Mug + Rug <= 2" - not part of the objective.
- Never equate a category (a position, a day, an hour) to a Boolean condition, and never invent a
  restriction the text does not state. Leave nothing out: every numbered clause AND every fact in
  the opening sentences - a budget, a limit on how many items, which values are in play - becomes
  a constraint of its own, not something the objective takes care of.

Objective - only when the puzzle asks to maximise or minimise
- One expression over the weighted sum, each stated weight used exactly once, no invented weights
  and no helper variables. Omit "objective" entirely otherwise.

{ir_spec}

Example:
Problem: Three trains leave from platforms 1, 2 and 3: an intercity, a regional and an express, one per platform. The express does not leave from platform 1. The regional leaves from a platform with a lower number than the intercity. The intercity does not leave from platform 3.
IR:
{{
  "variables": {{"Intercity": [1, 3], "Regional": [1, 3], "Express": [1, 3]}},
  "all_different": [["Intercity", "Regional", "Express"]],
  "constraints": ["Express != 1", "Regional < Intercity", "Intercity != 3"]
}}

Example (choosing items under a budget):
Problem: "Pick at most two of three items: Mug (cost 3, value 5), Lamp (cost 4, value 6) and
Rug (cost 2, value 4). The budget is 5. The Lamp may be built only if the Rug is."
IR:
{{
  "variables": {{"Mug": [0, 1], "Lamp": [0, 1], "Rug": [0, 1]}},
  "constraints": ["3*Mug + 4*Lamp + 2*Rug <= 5", "Mug + Lamp + Rug <= 2", "Lamp <= Rug"],
  "objective": {{"maximize": "5*Mug + 6*Lamp + 4*Rug"}}
}}

Puzzle:
{text}
{feedback}"""


def build_retry_feedback(ir: dict, errors: list[str]) -> str:
    instructions = [
        "Correct every error below.",
        "Return one compact, complete JSON object.",
        "Re-read the original puzzle. Preserve its meaning.",
    ]

    if any("Impossible all_different group" in e for e in errors):
        instructions.append(
            "MANDATORY CORRECTION: The previous all_different group is "
            "mathematically impossible. Re-read the original puzzle. "
            "If the puzzle does not explicitly require these variables "
            "to have pairwise-distinct values, set all_different to []. "
            "Do not repeat the impossible group. Do not add an objective "
            "unless the original puzzle requires optimization."
        )

    if any("objective" in e.lower() for e in errors):
        instructions.append(
            "Write the objective once, using only the weights in the "
            "original puzzle. For binary variables use weight * Variable. "
            "Include each weighted feature once; do not repeat terms or "
            "invent weights. Omit objective unless optimization is required."
        )

    instructions.append("Validation errors:\n- " + "\n- ".join(errors))
    return "\n".join(instructions)


def build_status_feedback(ir: dict, status: str) -> str:
    if status == "none":
        why = (
            "no assignment satisfies every constraint at once, so at least one constraint is "
            "stronger than what the puzzle says or contradicts another one."
        )
        fix = (
            "Re-read each clue and encode exactly what it states: no invented restriction, no "
            "extra condition, no clue dropped, and nothing stronger than the wording allows."
        )
    else:
        why = (
            "more than one assignment satisfies every constraint, so at least one clue is "
            "missing or encoded too loosely."
        )
        fix = (
            "Re-read every clue and encode each one exactly once. Each constraint must admit "
            "exactly the cases its clue describes: never join two possible readings of one "
            "clue with \"or\", never weaken a strict relation into a non-strict one, and never "
            "leave a clue out. If the puzzle assigns each value exactly once, that is an "
            "all_different group over those variables."
        )
    return (
        f"Your previous encoding was:\n{json.dumps(ir)}\n"
        f"The solver checked it and reported status \"{status}\": {why}\n"
        f"{fix}"
    )


def formalize(problem: dict, feedback: str | tuple[dict, list[str]] | None = None, temperature: float = 0.0) -> dict:
    fb = ""
    if isinstance(feedback, str):
        fb = f"\n{feedback}\nFollow the correction instructions exactly.\n"
    elif feedback is not None:
        prev, why = feedback
        fb = (
            f"\nYour previous encoding was:\n{json.dumps(prev)}\n"
            f"It was rejected because:\n{build_retry_feedback(prev, why)}\n"
            "Follow the correction instructions exactly.\n"
        )
    prompt = FORMALIZER_PROMPT.format(ir_spec=IR_SPEC, text=problem["text"], feedback=fb)
    parsed, raw = llm.ask_json(prompt, IRModel, system=SYSTEM, temperature=temperature)
    return parsed.model_dump(exclude_none=True)


def _validate_ir_variables(ir: dict, errors: list[str]) -> None:
    if "variables" not in ir:
        errors.append('IR must contain "variables" field')
        return

    for var_name, domain_or_values in ir["variables"].items():
        if not isinstance(var_name, str) or not VAR_NAME_PATTERN.fullmatch(var_name):
            errors.append(f"Invalid variable name: {var_name!r}")
            continue

        if isinstance(domain_or_values, list):
            if isinstance(domain_or_values, list) and len(domain_or_values) == 1 and type(domain_or_values[0]) is int:
                domain_or_values.append(domain_or_values[0])

            if len(domain_or_values) != 2 \
                    or not isinstance(domain_or_values[0], int) \
                    or not isinstance(domain_or_values[1], int):
                errors.append(
                    f'Variable "{var_name}" has invalid domain: {domain_or_values}, '
                    f"must be an array of two numbers in format [low, high]"
                )
                continue
            lo, hi = domain_or_values
            if lo > hi:
                errors.append(f'Variable "{var_name}" has invalid domain: {domain_or_values}, lo must be less than hi')
        elif isinstance(domain_or_values, dict):
            if "values" not in domain_or_values \
                    or not all(isinstance(value, int) for value in domain_or_values["values"]):
                errors.append(
                    f'Variable "{var_name}" has invalid values: {domain_or_values}, '
                    f'must be an object in format {{"values": [v1, v2, ...]}}'
                )
        else:
            errors.append(f'Variable "{var_name}" has invalid domain or values: {domain_or_values}')


def _validate_ir_all_different(ir: dict, errors: list[str]) -> None:
    variables = ir.get("variables")
    if not isinstance(variables, dict):
        return

    all_diff = ir.get("all_different", [])
    if not isinstance(all_diff, list):
        errors.append('"all_different" must be a list of lists of variable names in format [["var1", "var2"], ...]')
        return

    for group in all_diff:
        if not isinstance(group, list) or not all(isinstance(name, str) for name in group):
            errors.append('"all_different" must contain lists of variable names')
            continue

        if len(group) < 2:
            errors.append(f"Invalid all_different group {group}: a group must contain at least two variables")
            continue

        if len(set(group)) != len(group):
            errors.append(f"Invalid all_different group {group}: duplicate variable names")
            continue

        unknown = [name for name in group if name not in variables]
        if unknown:
            errors.append(f"all_different contains unknown variables: {unknown}")
            continue

        possible_values = set()
        domains_valid = True

        for name in group:
            domain = variables[name]

            if (
                isinstance(domain, list)
                and len(domain) == 2
                and all(type(v) is int for v in domain)
                and domain[0] <= domain[1]
            ):
                lo, hi = domain
                possible_values.update(range(lo, hi + 1))

            elif (
                isinstance(domain, dict)
                and isinstance(domain.get("values"), list)
                and domain["values"]
                and all(type(v) is int for v in domain["values"])
            ):
                possible_values.update(domain["values"])

            else:
                domains_valid = False

        if domains_valid and len(group) > len(possible_values):
            errors.append(
                f"Impossible all_different group {group}: "
                f"{len(group)} variables have only "
                f"{len(possible_values)} possible values"
            )


def _expression_root_error(expr: str, *, expect_bool: bool) -> str | None:
    try:
        root = ast.parse(expr, mode="eval").body
    except (SyntaxError, TypeError):
        return None

    is_not = isinstance(root, ast.UnaryOp) and isinstance(root.op, ast.Not)
    is_boolean = isinstance(root, (ast.Compare, ast.BoolOp)) or is_not

    if expect_bool and not is_boolean:
        return "constraint must be a Boolean condition, such as 'A == 1' or '(A == 1) or (B == 1)'"

    if not expect_bool and is_boolean:
        return "objective must be a numeric expression, not a Boolean condition; maximize or minimize the numeric expression directly"

    return None


def _validate_ir_objective(ir: dict, errors: list[str]) -> None:
    if "objective" not in ir:
        return

    obj = ir["objective"]
    if not isinstance(obj, dict) or not ("maximize" in obj or "minimize" in obj):
        errors.append(
            '"objective" must be an object '
            'in format {"maximise": "<expression>"} or {"minimize": "<expression>"}'
        )
        return

    expr = obj["maximize"] if "maximize" in obj else obj["minimize"]

    root_error = _expression_root_error(expr, expect_bool=False)
    if root_error:
        errors.append(f"Invalid objective: {root_error}")
        return

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
        errors.append(f'"constraints" must be an array in format ["expression", ...]')
        return

    conv = ir_solver._Conv({var_name: 0 for var_name in ir["variables"]})

    uniq_constraints = set()

    for constraint in constraints:
        if constraint in uniq_constraints:
            errors.append(f'Duplicate constraint: "{constraint}"')
            continue

        uniq_constraints.add(constraint)

        root_error = _expression_root_error(constraint, expect_bool=True)
        if root_error:
            errors.append(f'Invalid constraint "{constraint}": {root_error}')
            continue

        try:
            conv(ast.parse(constraint, mode="eval"))
        except (ValueError, SyntaxError) as e:
            errors.append(f'Error in constraint "{constraint}": {e}')


def _sanitize_names(ir: dict) -> None:
    variables = ir.get("variables")
    if not isinstance(variables, dict):
        return

    taken = set(variables)
    mapping: dict[str, str] = {}
    for name in list(variables):
        if isinstance(name, str) and VAR_NAME_PATTERN.fullmatch(name):
            continue
        base = re.sub(r"\W", "_", "_" + name if name[:1].isdigit() else name) or "_"
        new_name, i = base, 1
        while new_name in taken:
            i += 1
            new_name = f"{base}_{i}"
        taken.add(new_name)
        mapping[name] = new_name

    if not mapping:
        return

    def rewrite(expr: str) -> str:
        if not isinstance(expr, str):
            return expr
        for old, new in mapping.items():
            if old:
                expr = re.sub(rf"(?<!\w){re.escape(old)}(?!\w)" , new, expr)
        return expr

    ir["variables"] = {mapping.get(k, k): v for k, v in variables.items()}
    ir["constraints"] = [rewrite(c) for c in ir.get("constraints", [])]
    if isinstance(ir.get("all_different"), list):
        ir["all_different"] = [[rewrite(n) for n in g] if isinstance(g, list) else g for g in ir["all_different"]]
    if isinstance(ir.get("objective"), dict) and ir["objective"]:
        (obj, expression), = ir["objective"].items()
        ir["objective"] = {obj: rewrite(expression)}


def _repair_all_different(ir: dict) -> None:
    groups = ir.get("all_different")
    if not isinstance(groups, list):
        return
    ir["all_different"] = list(filter(lambda g: len(g) >= 2, map(set, groups)))


def validate_ir(ir: dict) -> list:
    _sanitize_names(ir)
    _repair_all_different(ir)

    errors = []

    _validate_ir_variables(ir, errors)
    _validate_ir_all_different(ir, errors)
    _validate_ir_objective(ir, errors)
    _validate_ir_constraints(ir, errors)

    if not errors:
        try:
            ir_solver.build(ir)
        except (ValueError, SyntaxError, KeyError, TypeError, IndexError) as e:
            err_msg = str(e)
            errors.append(f"Invalid expression or syntax: {err_msg}")

    return errors


def run_solver(ir: dict) -> dict:
    result = ir_solver.solve(ir, limit=2)
    solutions = result.get("solutions", [])

    if not solutions:
        result["status"] = "none"
        return result

    if len(solutions) >= 2:
        result["status"] = "multiple"
        return result

    if ir.get("objective") is not None:
        return result

    first = solutions[0]
    if not first:
        return result

    blocked_ir = deepcopy(ir)
    terms = [
        f"({name} == {value})"
        for name, value in first.items()
    ]
    blocked_ir.setdefault("constraints", []).append(f"not ({' and '.join(terms)})")

    alternative = ir_solver.solve(blocked_ir, limit=1)

    if alternative.get("solutions"):
        result["status"] = "multiple"
    else:
        result["status"] = "unique"

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
    prompt = PRESENT_PROMPT.format(
        text=problem["text"],
        ir=json.dumps(ir, ensure_ascii=False),
        assignment=json.dumps(result["solutions"][0], ensure_ascii=False),
        answer_format=problem["answer_format"],
    )
    parsed, raw = llm.ask_json(prompt, Answer)
    return Answer(status=status, solution=parsed.solution, objective=result.get("objective")).model_dump()


def solve(problem: dict) -> dict:
    """Returns {"answer": <Answer as dict>, "trace": {...}}. Raise HybridError if no answer can be produced."""
    trace = {"attempts": []}

    ir = formalize(problem)
    errors = validate_ir(ir)
    trace["attempts"].append({"ir": ir, "errors": errors})

    if errors:
        ir = formalize(problem, feedback=(ir, errors))
        new_errors = validate_ir(ir)
        trace["attempts"].append({"ir": ir, "errors": new_errors})
        if new_errors:
            raise HybridError("invalid IR after retry: " + "; ".join(new_errors))

    try:
        result = run_solver(ir)
    except ValueError as e:                      # the solver rejected the IR
        raise HybridError(f"solver rejected the IR: {e}") from e

    if result["status"] in ("none", "multiple"):
        new_ir = None
        new_result = None

        try:
            new_ir = formalize(problem, feedback=build_status_feedback(ir, result["status"]), temperature=RETRY_TEMPERATURE)
            new_errors = validate_ir(new_ir)
            if not new_errors:
                new_result = run_solver(new_ir)
        except (llm.LLMError, ValueError) as e:
            new_errors = [f"status retry failed: {e}"]
        trace["attempts"].append({"ir": new_ir, "errors": new_errors})
        if new_result is not None and STATUS_RANK[new_result["status"]] > STATUS_RANK[result["status"]]:
            ir, result = new_ir, new_result

    trace["solver"] = {
        "status": result["status"],
        "n_solutions": len(result["solutions"]),
        "objective": result.get("objective"),
        "first_solution": result["solutions"][0] if result["solutions"] else None
    }

    answer = present(problem, ir, result)
    return {"answer": answer, "trace": trace}
