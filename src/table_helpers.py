import os

import numpy as np


# Fixed decimals. A leading "-" becomes a true minus sign, in text and in math.
def fmt(x, d=4):
    arr = np.asarray(x, dtype=float)
    scalar = arr.ndim == 0
    out = []
    for v in np.atleast_1d(arr):
        if not np.isfinite(v):
            out.append("NA")
            continue
        text = f"{v:.{int(d)}f}"
        if text.startswith("-"):
            text = "\\ensuremath{-}" + text[1:]
        out.append(text)
    return out[0] if scalar else out


# p-values: fixed notation, or a x 10^b wrapped so it works in text and in math.
def fmtp(p, d=4):
    arr = np.asarray(p, dtype=float)
    scalar = arr.ndim == 0
    out = []
    for v in np.atleast_1d(arr):
        if not np.isfinite(v):
            out.append("NA")
            continue
        if v >= 10 ** (-d):
            out.append(f"{v:.{int(d)}f}")
            continue
        b = int(np.floor(np.log10(v)))
        a = v / 10**b
        out.append(
            f"\\ensuremath{{{a:.2f}\\times10^{{{b}}}}}"
        )
    return out[0] if scalar else out


# Collect macros and write them to one .tex file.
_macros = {}


def clear_macros():
    _macros.clear()


def set_macro(name, value):
    _macros[name] = value


def _as_tex(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return repr(number)
    return str(value)


def write_macros(file):
    os.makedirs(os.path.dirname(file) or ".", exist_ok=True)
    lines = [
        "%% Generated automatically by the Python scripts -- do not edit by hand"
    ]
    for name in sorted(_macros):
        lines.append(f"\\newcommand{{\\{name}}}{{{_as_tex(_macros[name])}}}")
    with open(file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
