from __future__ import annotations

import re
from dataclasses import dataclass


class MacroFrontendError(RuntimeError):
    """Raised when a selected TeX figure macro cannot be materialized safely."""


@dataclass(frozen=True)
class MacroDefinition:
    name: str
    arg_count: int
    body: str
    source_line: int
    default_first_arg: str | None = None


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _skip_space(text: str, offset: int) -> int:
    while offset < len(text) and text[offset].isspace():
        offset += 1
    return offset


def _read_balanced(
    text: str,
    offset: int,
    opener: str,
    closer: str,
) -> tuple[str, int] | None:
    offset = _skip_space(text, offset)
    if offset >= len(text) or text[offset] != opener:
        return None
    depth = 0
    escaped = False
    for index in range(offset, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[offset + 1 : index], index + 1
    return None


def extract_command_macros(text: str) -> dict[str, MacroDefinition]:
    """Read the ordinary ``newcommand`` family used by the handout figures.

    This is deliberately not a TeX engine.  It only materializes braced macro
    templates so the native TikZ compiler sees the concrete ``tikzpicture``
    selected by the caller.  TikZ/geometry primitives remain opaque and are
    interpreted by the semantic compiler instead.
    """

    command_re = re.compile(r"\\(?:newcommand|renewcommand|providecommand)\b")
    definitions: dict[str, MacroDefinition] = {}
    cursor = 0
    while match := command_re.search(text, cursor):
        name_group = _read_balanced(text, match.end(), "{", "}")
        if name_group is None:
            cursor = match.end()
            continue
        raw_name, after_name = name_group
        name_match = re.fullmatch(r"\\([A-Za-z@]+)", raw_name.strip())
        if name_match is None:
            cursor = after_name
            continue

        arg_count = 0
        default_first_arg: str | None = None
        after_signature = after_name
        count_group = _read_balanced(text, after_signature, "[", "]")
        if count_group is not None and count_group[0].strip().isdigit():
            arg_count = int(count_group[0].strip())
            after_signature = count_group[1]
            default_group = _read_balanced(text, after_signature, "[", "]")
            if default_group is not None:
                default_first_arg = default_group[0]
                after_signature = default_group[1]

        body_group = _read_balanced(text, after_signature, "{", "}")
        if body_group is None:
            cursor = after_signature
            continue
        body, cursor = body_group
        definitions[name_match.group(1)] = MacroDefinition(
            name=name_match.group(1),
            arg_count=arg_count,
            body=body,
            source_line=_line_number(text, match.start()),
            default_first_arg=default_first_arg,
        )
    return definitions


def _expand_pass(
    text: str,
    definitions: dict[str, MacroDefinition],
    opaque_names: set[str],
) -> tuple[str, bool]:
    command_re = re.compile(r"\\(?P<name>[A-Za-z@]+)(?![A-Za-z@])")
    output: list[str] = []
    cursor = 0
    changed = False
    while match := command_re.search(text, cursor):
        name = match.group("name")
        definition = definitions.get(name)
        if definition is None or name in opaque_names:
            output.append(text[cursor : match.end()])
            cursor = match.end()
            continue

        argument_cursor = match.end()
        arguments: list[str] = []
        if definition.default_first_arg is not None:
            optional = _read_balanced(text, argument_cursor, "[", "]")
            if optional is None:
                arguments.append(definition.default_first_arg)
            else:
                arguments.append(optional[0])
                argument_cursor = optional[1]

        required_count = definition.arg_count - len(arguments)
        valid = True
        for _ in range(required_count):
            group = _read_balanced(text, argument_cursor, "{", "}")
            if group is None:
                valid = False
                break
            arguments.append(group[0])
            argument_cursor = group[1]
        if not valid:
            output.append(text[cursor : match.end()])
            cursor = match.end()
            continue

        replacement = definition.body
        for index, argument in enumerate(arguments, start=1):
            replacement = replacement.replace(f"#{index}", argument)
        output.append(text[cursor : match.start()])
        output.append(replacement)
        cursor = argument_cursor
        changed = True

    output.append(text[cursor:])
    return "".join(output), changed


def materialize_entry_macro(
    text: str,
    entry_macro: str,
    *,
    max_passes: int = 24,
    opaque_names: set[str] | None = None,
) -> tuple[str, int]:
    """Expand one named figure entry into concrete TikZ source.

    The returned line is the definition line of the selected entry.  It is a
    coarse but useful source anchor for generated picture reports.
    """

    clean_name = entry_macro.strip().lstrip("\\")
    definitions = extract_command_macros(text)
    if clean_name not in definitions:
        raise MacroFrontendError(f"Unknown figure macro: \\{clean_name}")
    if definitions[clean_name].arg_count:
        raise MacroFrontendError(
            f"Entry macro \\{clean_name} requires "
            f"{definitions[clean_name].arg_count} argument(s); select a concrete wrapper"
        )

    opaque = {
        "defPointShift",
        "defPointOnSpanPlane",
    }
    if opaque_names:
        opaque.update(opaque_names)

    expanded = rf"\{clean_name}"
    for _ in range(max_passes):
        expanded, changed = _expand_pass(expanded, definitions, opaque)
        if not changed:
            break
    else:
        raise MacroFrontendError(
            f"Macro expansion for \\{clean_name} exceeded {max_passes} passes"
        )

    unresolved_parameters = sorted(set(re.findall(r"#[1-9]", expanded)))
    if unresolved_parameters:
        raise MacroFrontendError(
            f"Unresolved macro parameters in \\{clean_name}: "
            + ", ".join(unresolved_parameters)
        )
    return expanded, definitions[clean_name].source_line
