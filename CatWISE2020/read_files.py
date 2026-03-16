import gzip
import numpy as np
import astropy.units as u
from astropy.table import Table, MaskedColumn

def _open_gzip(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _parse_ipac_type(ipac_type):
    t = ipac_type.strip().lower()

    # IPAC short and long type names
    if t in {"r", "real", "float", "double", "d"}:
        return "float"
    if t in {"i", "int", "integer", "long"}:
        return "int"
    if t in {"char", "c", "string"}:
        return "str"

    # fallback
    return "str"


def _parse_ipac_unit(unit_str):
    unit_str = unit_str.strip()

    # IPAC placeholders meaning "no useful unit"
    if unit_str in {"", "--", "-", "null"}:
        return None

    # Some common catalog shorthands
    replacements = {
        "asec": u.arcsec,
        "asecpyr": u.arcsec / u.yr,
        "deg": u.deg,
        "pix": u.pix,
        "mag": u.mag,
        "days": u.day,
        "dn": None,   # data number; leave as metadata-free unless you want a custom unit
        "%": u.percent,
    }

    if unit_str in replacements:
        return replacements[unit_str]

    try:
        return u.Unit(unit_str)
    except Exception:
        return None


def read_ipac_columns(path, wanted):
    with _open_gzip(path) as f:
        # find first header line with column names
        for line in f:
            if line.lstrip().startswith("|"):
                name_line = line.rstrip("\n")
                break
        else:
            raise ValueError("No IPAC header found")

        type_line = next(f).rstrip("\n")
        unit_line = next(f).rstrip("\n")
        null_line = next(f).rstrip("\n")

        pipes = [i for i, ch in enumerate(name_line) if ch == "|"]

        # read all column specs from header
        spec_map = {}
        for left, right in zip(pipes[:-1], pipes[1:]):
            start, end = left + 1, right

            name = name_line[start:end].strip()
            if not name:
                continue

            ipac_type = type_line[start:end].strip()
            unit_str = unit_line[start:end].strip()
            null_str = null_line[start:end].strip()

            spec_map[name] = {
                "name": name,
                "kind": _parse_ipac_type(ipac_type),
                "ipac_type": ipac_type,
                "unit_str": unit_str,
                "unit": _parse_ipac_unit(unit_str),
                "null": null_str,
                "start": start,
                "end": end,
            }

        missing = [col for col in wanted if col not in spec_map]
        if missing:
            raise KeyError(f"Columns not found: {missing}")

        # preserve requested order
        specs = [spec_map[col] for col in wanted]

        values = {col: [] for col in wanted}
        masks = {col: [] for col in wanted}

        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith("\\"):
                continue

            for spec in specs:
                name = spec["name"]
                raw = line[spec["start"]:spec["end"]].strip()
                null_token = spec["null"]

                is_null = (raw == "") or (null_token not in {"", "--"} and raw == null_token)
                masks[name].append(is_null)

                if is_null:
                    if spec["kind"] == "int":
                        values[name].append(0)
                    elif spec["kind"] == "float":
                        values[name].append(np.nan)
                    else:
                        values[name].append("")
                else:
                    if spec["kind"] == "int":
                        values[name].append(int(raw))
                    elif spec["kind"] == "float":
                        values[name].append(float(raw))
                    else:
                        values[name].append(raw)

    out = Table(masked=True)

    for spec in specs:
        name = spec["name"]
        kind = spec["kind"]
        mask = np.array(masks[name], dtype=bool)

        if kind == "int":
            data = np.array(values[name], dtype=np.int64)
        elif kind == "float":
            data = np.array(values[name], dtype=np.float64)
        else:
            data = np.array(values[name], dtype=object)

        col = MaskedColumn(data, mask=mask, name=name)

        if spec["unit"] is not None:
            col.unit = spec["unit"]

        out.add_column(col)

    return out
