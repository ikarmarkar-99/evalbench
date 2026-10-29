"""Orchestrator for converting SQL queries between database dialects.

This module contains:
  - Plain-SQL tie-breaker helpers (`_split_top_level_commas`,
    `_strip_select_alias`, `_parse_schema_columns`, `_inject_window_tiebreakers`,
    and `_inject_deterministic_tiebreakers`) that add shared tie-breaker columns
    to `ORDER BY` clauses when a query has ties, so both the source and
    converted queries can be compared across 100% of columns without dropping
    any columns.
  - `QueryConversionOrchestrator`: Runs the iterative conversion loop against
    the source and destination databases.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import re
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple
from .query_converters.base import (
    ConversionResult,
    ConversionStatus,
    QueryConverter,
    QueryResult,
)
from .query_executor import QueryExecutor
from .utils.llm import gemini_inference


# SQL keywords that can follow a table name in a `FROM` clause (so the parser
# knows `FROM courses WHERE ...` has no table nickname, rather than treating
# `WHERE` as the nickname of `courses`).
_SQL_KEYWORDS = frozenset({
    "WHERE",
    "JOIN",
    "INNER",
    "LEFT",
    "RIGHT",
    "FULL",
    "CROSS",
    "ON",
    "GROUP",
    "ORDER",
    "LIMIT",
    "HAVING",
    "UNION",
    "EXCEPT",
    "INTERSECT",
    "WINDOW",
    "OFFSET",
    "FETCH",
    "FOR",
    "AS",
    "NATURAL",
    "USING",
    "OUTER",
    "SELECT",
    "FROM",
    "WITH",
})


def _split_top_level_commas(expr_list: str) -> List[str]:
    """Split a comma-separated SQL list while ignoring commas inside parentheses or quotes.

    Example:
        Input:  `"a, COALESCE(b, c), 'x, y'"`
        Output: `["a", "COALESCE(b, c)", "'x, y'"]`

    Args:
        expr_list: Comma-separated SQL text (such as the columns after `SELECT`
            or `ORDER BY`).

    Returns:
        List of trimmed SQL expressions.
    """
    parts = []
    current = []
    depth = 0
    in_single_quote = False
    in_double_quote = False
    # Walk character by character, tracking whether we are inside quotes or parentheses.
    for ch in expr_list:
        if ch == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif not in_single_quote and not in_double_quote:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                # Only split when we see a comma at the outermost level (depth == 0).
                parts.append("".join(current).strip())
                current = []
                continue
        current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _mask_parens_and_strings(sql: str) -> str:
    """Replace text inside parentheses and single quotes with spaces of the same length.

    This makes it safe to search for outer keywords like `FROM`, `WHERE`, or
    `ORDER BY` without accidentally matching a `FROM` or `ORDER BY` that is
    hidden inside a subquery `(...)` or a quoted string `'...'`. Because every
    replaced character becomes 1 space, character positions in the returned
    string still match the exact same positions in `sql`.

    Example:
        Input:  `"SELECT (SELECT x FROM t1) AS c FROM t2 WHERE note = 'FROM'"`
        Output: `"SELECT (                ) AS c FROM t2 WHERE note =       "`

    Args:
        sql: Raw SQL query or clause string.

    Returns:
        Same-length string with nested parentheses and string literals blanked out.
    """
    out = []
    depth = 0
    in_sq = False
    in_dq = False
    for ch in sql:
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            out.append(" ")
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
            out.append(ch)
        elif in_sq:
            out.append(" ")
        else:
            if ch == "(":
                out.append("(" if depth == 0 else " ")
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
                out.append(")" if depth == 0 else " ")
            elif depth > 0:
                out.append(" ")
            else:
                out.append(ch)
    return "".join(out)


def _find_matching_paren(text: str, open_idx: int) -> int:
    """Find the position of the closing `)` that matches the `(` at `open_idx`.

    Example:
        Input:  `text = "OVER (ORDER BY UPPER(dept))", open_idx = 5`
        Output: `26` (the index of the outer `)`, skipping the inner `(dept)`)

    Args:
        text: SQL text containing the parenthesized expression.
        open_idx: Character index of the opening `(`.

    Returns:
        Character index of the matching `)`, or `-1` if no matching `)` is found.
    """
    depth = 0
    in_sq = False
    in_dq = False
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif not in_sq and not in_dq:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
    return -1


def _strip_select_alias(expr: str) -> str:
    """Remove a trailing `AS column_alias` rename from a `SELECT` column expression.

    Example:
        Input:  `"s.subject_title AS title"`
        Output: `"s.subject_title"`

    Args:
        expr: A single column expression from a `SELECT` list.

    Returns:
        The column expression without its trailing `AS ...` alias.
    """
    masked = _mask_parens_and_strings(expr)
    # Match a trailing `AS identifier` at the end of the expression (outside parentheses).
    as_m = re.search(
        r"\s+\bAS\s+[`\"]?\w+[`\"]?\s*$", masked, flags=re.IGNORECASE
    )
    if as_m:
        return expr[: as_m.start()].strip()
    return expr.strip()


def _parse_schema_columns(
    schema_ddl: str,
) -> Tuple[Dict[str, List[Tuple[str, bool]]], Set[str]]:
    """Read the database table definitions to find each table's columns and which columns are text.

    Example:
        Input:  `"CREATE TABLE courses (id INT, title VARCHAR(100));"`
        Output: `({"courses": [("id", False), ("title", True)]}, {"title"})`

    Args:
        schema_ddl: Optional database table definition text (`CREATE TABLE ...`)
            listing each table's column names and types (text vs. number/date).

    Returns:
        Tuple of `(tables, pure_text_cols)` where `tables` maps each lowercase
        table name to `[(column_name, is_text_column), ...]` and `pure_text_cols`
        is the set of column names that are always text across the database.
    """
    tables: Dict[str, List[Tuple[str, bool]]] = {}
    if not schema_ddl:
        return tables, set()
    # Match each `CREATE TABLE table_name ( ... )` block in the schema file.
    create_re = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"]?(\w+)[`\"]?\s*\((.*?)\)\s*(?:ENGINE|;)",
        re.IGNORECASE | re.DOTALL,
    )
    # Lines inside `CREATE TABLE (...)` that start with these words are table
    # constraints (like `PRIMARY KEY (...)`), not column definitions.
    skip_starts = {
        "primary",
        "foreign",
        "unique",
        "key",
        "index",
        "constraint",
        "check",
    }
    text_col_names: Set[str] = set()
    nontext_col_names: Set[str] = set()
    for m in create_re.finditer(schema_ddl):
        tname = m.group(1).lower()
        body = m.group(2)
        cols: List[Tuple[str, bool]] = []
        for part in _split_top_level_commas(body):
            if not part:
                continue
            first_word = part.split()[0].strip('`"').lower()
            if first_word in skip_starts or part.upper().startswith("CONSTRAINT "):
                continue
            # Extract `column_name DATA_TYPE` from the start of the column definition line.
            col_m = re.match(r"^[`\"]?(\w+)[`\"]?\s+([a-zA-Z0-9_]+)", part)
            if col_m:
                cname = col_m.group(1).lower()
                ctype = col_m.group(2).lower()
                is_text = any(
                    t in ctype for t in ("char", "text", "string", "enum", "set")
                )
                cols.append((cname, is_text))
                if is_text:
                    text_col_names.add(cname)
                else:
                    nontext_col_names.add(cname)
        if cols:
            tables[tname] = cols
    return tables, text_col_names - nontext_col_names


def _extract_window_func_args(item: str) -> List[str]:
    """Find the column being averaged, summed, or looked up inside a window function.

    Example:
        Input:  `"AVG(s.total_units) OVER (ORDER BY s.term_code) AS moving_avg"`
        Output: `["s.total_units"]`

    Args:
        item: A single column expression from a `SELECT` list.

    Returns:
        List of column/expression arguments inside `MIN`, `MAX`, `AVG`, `SUM`,
        `LAG`, `LEAD`, `FIRST_VALUE`, or `LAST_VALUE` before `OVER (...)`.
    """
    args = []
    # Match window functions that take a column expression as their first argument.
    fn_re = re.compile(
        r"\b(?:MIN|MAX|AVG|SUM|LAG|LEAD|FIRST_VALUE|LAST_VALUE)\s*\(",
        re.IGNORECASE,
    )
    for m in fn_re.finditer(item):
        open_p = m.end() - 1
        close_p = _find_matching_paren(item, open_p)
        if close_p == -1:
            continue
        after = item[close_p + 1 :].lstrip()
        # Only treat it as a window function if the closing `)` is followed by `OVER`.
        if not re.match(r"^OVER\b", after, flags=re.IGNORECASE):
            continue
        inner = item[open_p + 1 : close_p].strip()
        if not inner or inner == "*":
            continue
        first_arg = _split_top_level_commas(inner)[0].strip()
        first_arg = re.sub(
            r"^\s*DISTINCT\s+", "", first_arg, flags=re.IGNORECASE
        ).strip()
        if first_arg and "*" not in first_arg:
            args.append(first_arg)
    return args


def _is_expr_text_col(
    base_expr: str,
    alias_to_table: Dict[str, str],
    schema_tables: Dict[str, List[Tuple[str, bool]]],
    pure_text_cols: Set[str],
) -> bool:
    """Check whether a SQL column reference points to a text column (rather than a number or date).

    Example:
        If table `courses` (nicknamed `c`) has `id INT` and `title VARCHAR(100)`:
        - `_is_expr_text_col('c.title', ...)` -> `True`
        - `_is_expr_text_col('c.id', ...)`    -> `False`

    Args:
        base_expr: Column reference (such as `t1.subject_id` or `department_name`).
        alias_to_table: Mapping from query table nicknames (like `t1`) to real table names.
        schema_tables: Table column definitions from `_parse_schema_columns`.
        pure_text_cols: Column names that are always text across the database.

    Returns:
        True if the column is a text column.
    """
    # Match an optional `table_alias.` prefix followed by `column_name`.
    m = re.match(
        r"^(?:([`\"]?\w+[`\"]?)\.)?([`\"]?\w+[`\"]?)$", base_expr.strip()
    )
    if not m:
        return False
    qual = m.group(1).strip('`"').lower() if m.group(1) else None
    col = m.group(2).strip('`"').lower()
    # If the column has a table prefix (like `t1.subject_id`), look up `t1`'s table in the schema.
    if qual and qual in alias_to_table:
        tname = alias_to_table[qual]
        for cname, is_t in schema_tables.get(tname, []):
            if cname == col:
                return is_t
        return False
    # Otherwise, check if that column name is always text across the schema.
    if col in pure_text_cols:
        return True
    return col.startswith("hist_") and col[5:] in pure_text_cols


def _format_tie_expr(
    expr: str,
    dialect: str,
    is_text: bool,
    direction: str = "ASC",
) -> str:
    """Format a column for `ORDER BY` so MySQL and PostgreSQL sort values in the exact same order.

    MySQL and PostgreSQL have three default sorting differences:
      1. Empty (`NULL`) values: MySQL puts `NULL`s first on `ASC` and last on `DESC`;
         PostgreSQL does the opposite unless `NULLS FIRST` / `NULLS LAST` is specified.
      2. Uppercase vs. lowercase text: MySQL sorts case-insensitively by default,
         while PostgreSQL sorts capital letters before lowercase letters unless
         `UPPER(...)` is used.
      3. Hyphens vs. spaces in text codes: MySQL's default Unicode sorting treats
         `'-'` and `' '` similarly before uppercase letters, so normalizing `'-'`
         to `' '` inside `UPPER(REPLACE(...))` keeps text codes aligned.

    Example (for a numeric column `c.id` in ascending order):
        - MySQL:      `"c.id IS NOT NULL ASC, c.id ASC"`
        - PostgreSQL: `'c."id" ASC NULLS FIRST'`

    Example (for a text column `c.title` in ascending order):
        - MySQL:      `"c.title IS NOT NULL ASC, CAST(UPPER(REPLACE(c.title, '-', ' ')) AS BINARY) ASC, CAST(c.title AS BINARY) ASC"`
        - PostgreSQL: `'UPPER(REPLACE(c."title", \'-\', \' \')) COLLATE "C" ASC NULLS FIRST, c."title" COLLATE "C" ASC NULLS FIRST'`

    Args:
        expr: SQL column or expression to sort by.
        dialect: `"mysql"`, `"postgresql"`, or `""` (plain SQL for unit tests).
        is_text: True if `expr` is a text column (so `UPPER(...)` is safe to apply).
        direction: `"ASC"` (ascending) or `"DESC"` (descending).

    Returns:
        Formatted `ORDER BY` item string for the target database dialect.
    """
    # Strip any existing `ASC`, `DESC`, or `NULLS FIRST/LAST` suffix from `expr`.
    base = re.sub(
        r"\s*(?:\b(?:ASC|DESC)\b|\bNULLS\s+(?:FIRST|LAST)\b)\s*$",
        "",
        expr.strip(),
        flags=re.IGNORECASE,
    ).strip()
    base = re.sub(
        r"\s*(?:\b(?:ASC|DESC)\b|\bNULLS\s+(?:FIRST|LAST)\b)\s*$",
        "",
        base,
        flags=re.IGNORECASE,
    ).strip()
    if not dialect:
        return base if direction.upper() == "ASC" else f"{base} DESC"

    has_collate = "COLLATE" in base.upper()
    direction = direction.upper()
    if dialect == "mysql":
        if is_text and not has_collate:
            if direction == "ASC":
                return (
                    f"{base} IS NOT NULL ASC, "
                    f"CAST(UPPER(REPLACE({base}, '-', ' ')) AS BINARY) ASC, "
                    f"CAST({base} AS BINARY) ASC"
                )
            return (
                f"{base} IS NULL ASC, "
                f"CAST(UPPER(REPLACE({base}, '-', ' ')) AS BINARY) DESC, "
                f"CAST({base} AS BINARY) DESC"
            )
        if direction == "ASC":
            return f"{base} IS NOT NULL ASC, {base} ASC"
        return f"{base} IS NULL ASC, {base} DESC"

    # PostgreSQL formatting: match MySQL's NULL placement and case-insensitive text order.
    if is_text and not has_collate:
        if direction == "ASC":
            return (
                f'UPPER(REPLACE({base}, \'-\', \' \')) COLLATE "C" ASC NULLS FIRST, '
                f'{base} COLLATE "C" ASC NULLS FIRST'
            )
        return (
            f'UPPER(REPLACE({base}, \'-\', \' \')) COLLATE "C" DESC NULLS LAST, '
            f'{base} COLLATE "C" DESC NULLS LAST'
        )
    if direction == "ASC":
        return f"{base} ASC NULLS FIRST"
    return f"{base} DESC NULLS LAST"


def _extract_key_dir(expr: str) -> Tuple[str, str]:
    """Split an `ORDER BY` item into its column name and sort direction (`ASC` or `DESC`).

    Example:
        Input:  `'UPPER(t1."dept") COLLATE "C" DESC NULLS LAST'`
        Output: `('t1."dept"', 'DESC')`

    Args:
        expr: A single item from an `ORDER BY` clause.

    Returns:
        Tuple of `(column_expression, direction)` where `direction` is `'ASC'` or `'DESC'`.
    """
    s = expr.strip()
    # Check if the item ends with `ASC` or `DESC` (optionally followed by `NULLS FIRST/LAST`).
    dir_m = re.search(
        r"\b(ASC|DESC)\b(?:\s+NULLS\s+(?:FIRST|LAST))?\s*$",
        s,
        flags=re.IGNORECASE,
    )
    direction = dir_m.group(1).upper() if dir_m else "ASC"
    # Remove trailing `ASC`/`DESC`, `NULLS FIRST/LAST`, `COLLATE ...`, and `UPPER(...)`/`LOWER(...)`
    # wrappers so we get the clean underlying column expression.
    base = re.sub(
        r"\s*(?:\b(?:ASC|DESC)\b|\bNULLS\s+(?:FIRST|LAST)\b)\s*$",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    base = re.sub(
        r"\s*(?:\b(?:ASC|DESC)\b|\bNULLS\s+(?:FIRST|LAST)\b)\s*$",
        "",
        base,
        flags=re.IGNORECASE,
    ).strip()
    base = re.sub(
        r"\s+\bCOLLATE\s+(?:\"[^\"]+\"|'[^']+'|\w+)\s*$",
        "",
        base,
        flags=re.IGNORECASE,
    ).strip()
    lu_m = re.match(
        r"^(?:LOWER|UPPER)\s*\((.*)\)$", base, flags=re.IGNORECASE | re.DOTALL
    )
    if lu_m:
        base = lu_m.group(1).strip()
    return base, direction


def _extract_cte_columns(
    sql: str,
    pure_text_cols: Set[str],
) -> Dict[str, List[Tuple[str, bool]]]:
    """Find the output column names of each `WITH helper_table AS (SELECT ...)` block in `sql`.

    When a later part of the query writes `SELECT * FROM helper_table`, knowing
    the columns of `helper_table` lets us use them as tie-breakers.

    Example:
        Input:  `"WITH top_courses AS (SELECT dept, COUNT(*) AS cnt FROM c GROUP BY dept) SELECT * FROM top_courses"`
        Output: `{"top_courses": [("dept", True), ("cnt", False)]}`

    Args:
        sql: Full SQL query string.
        pure_text_cols: Set of column names known to be text columns.

    Returns:
        Mapping from lowercase helper table (CTE) name to `[(column_name, is_text), ...]`.
    """
    cte_map: Dict[str, List[Tuple[str, bool]]] = {}
    # Match `cte_name AS (` headers in `WITH ...` blocks.
    cte_re = re.compile(
        r"([`\"]?\w+[`\"]?)\s+AS\s+(?:(?:NOT\s+)?MATERIALIZED\s+)?\(",
        re.IGNORECASE,
    )
    for m in cte_re.finditer(sql):
        cname = m.group(1).strip('`"').lower()
        open_p = m.end() - 1
        close_p = _find_matching_paren(sql, open_p)
        if close_p == -1:
            continue
        inner = sql[open_p + 1 : close_p].strip()
        masked = _mask_parens_and_strings(inner)
        sel_m = re.search(
            r"^\s*SELECT\s+(?:DISTINCT\s+)?", masked, flags=re.IGNORECASE
        )
        from_m = re.search(r"\bFROM\b", masked, flags=re.IGNORECASE)
        if not sel_m or not from_m:
            continue
        s_body = inner[sel_m.end() : from_m.start()]
        cols: List[Tuple[str, bool]] = []
        # Extract each column's output name (either its `AS alias` or its bare column name).
        for item in _split_top_level_commas(s_body):
            item_s = item.strip()
            if "*" in item_s or ":=" in item_s:
                continue
            item_masked = _mask_parens_and_strings(item_s)
            as_m = re.search(
                r"\bAS\s+([`\"]?\w+[`\"]?)\s*$",
                item_masked,
                flags=re.IGNORECASE,
            )
            if as_m:
                out_col = item_s[as_m.start(1) : as_m.end(1)].strip('`"')
            else:
                tail_m = re.search(r"([`\"]?\w+[`\"]?)\s*$", item_s)
                out_col = tail_m.group(1).strip('`"') if tail_m else ""
            if out_col:
                col_l = out_col.lower()
                cols.append((out_col, col_l in pure_text_cols))
        if cols:
            cte_map[cname] = cols
    return cte_map


def _inject_tiebreakers_in_select(
    block_sql: str,
    dialect: str,
    schema_tables: Dict[str, List[Tuple[str, bool]]],
    pure_text_cols: Set[str],
    cte_tables: Optional[Dict[str, List[Tuple[str, bool]]]] = None,
) -> str:
    """Add shared tie-breaker columns to `OVER (... ORDER BY ...)` and `ORDER BY ... LIMIT` in one `SELECT` block.

    How it works:
      1. Collects the other columns in the `SELECT` clause and the columns of any
         tables in the `FROM` clause.
      2. Appends those columns as tie-breakers to any tie-sensitive window
         calculation (`ROW_NUMBER()`, `LAG()`, `LEAD()`, or a `ROWS BETWEEN`
         moving window), while leaving `RANK()` and `DENSE_RANK()` untouched
         because tied rows in `RANK()`/`DENSE_RANK()` are supposed to share the
         same rank number.
      3. Appends those columns as tie-breakers to any `ORDER BY ... LIMIT` clause
         at the end of the `SELECT` block.

    Example:
        Input:  `"SELECT dept, title, ROW_NUMBER() OVER (ORDER BY dept) AS rn FROM courses"`
        Output: `"SELECT dept, title, ROW_NUMBER() OVER (ORDER BY dept, dept, title) AS rn FROM courses"`

    Args:
        block_sql: A single `SELECT ... FROM ...` block.
        dialect: `"mysql"`, `"postgresql"`, or `""`.
        schema_tables: Table column definitions from `_parse_schema_columns`.
        pure_text_cols: Set of column names known to be text columns.
        cte_tables: Optional helper-table (CTE) column map from `_extract_cte_columns`.

    Returns:
        The `SELECT` block with shared tie-breaker columns added to its `ORDER BY` clauses.
    """
    cte_tables = cte_tables or {}
    masked = _mask_parens_and_strings(block_sql)
    # Locate the top-level `SELECT` and `FROM` keywords in this block.
    select_m = re.search(
        r"^\s*SELECT\s+(DISTINCT\s+)?", masked, flags=re.IGNORECASE
    )
    from_m = re.search(r"\bFROM\b", masked, flags=re.IGNORECASE)
    if not select_m or not from_m:
        return block_sql
    # Skip `UNION` blocks so each branch's column count and syntax stay untouched.
    if re.search(r"\bUNION\b", masked, flags=re.IGNORECASE):
        return block_sql

    is_distinct = bool(select_m.group(1))
    select_body = block_sql[select_m.end() : from_m.start()]
    after_from_masked = masked[from_m.end() :]
    after_from_raw = block_sql[from_m.end() :]
    has_group_by = bool(
        re.search(r"\bGROUP\s+BY\b", after_from_masked, flags=re.IGNORECASE)
    )

    # Isolate just the `FROM ... JOIN ...` portion (stopping before `WHERE`, `GROUP BY`, etc.).
    end_m = re.search(
        r"\b(?:WHERE|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|WINDOW)\b",
        after_from_masked,
        flags=re.IGNORECASE,
    )
    from_body_masked = (
        after_from_masked[: end_m.start()] if end_m else after_from_masked
    )
    alias_to_table: Dict[str, str] = {}
    table_key_exprs: List[Tuple[str, bool]] = []
    q = "`" if dialect == "mysql" else ('"' if dialect == "postgresql" else "")
    has_bare_star = bool(re.match(r"^\s*\*\s*$", select_body))

    # Find each table name and its optional nickname (alias) in `FROM table alias JOIN table2 alias2`.
    t_pat = re.compile(
        r"(?:^\s*|\bJOIN\s+)([`\"]?\w+[`\"]?)(?:\s+(?:AS\s+)?([`\"]?\w+[`\"]?))?",
        re.IGNORECASE,
    )
    for tm in t_pat.finditer(from_body_masked):
        raw_t = tm.group(1).strip('`"').lower()
        raw_a = (
            tm.group(2).strip('`"')
            if tm.group(2)
            else tm.group(1).strip('`"')
        )
        if raw_a.upper() in _SQL_KEYWORDS:
            raw_a = tm.group(1).strip('`"')
        if raw_t in schema_tables:
            alias_to_table[raw_a.lower()] = raw_t
            # When the query does not group or deduplicate rows (`GROUP BY` / `DISTINCT`),
            # we can also use the table's own columns from the schema as tie-breakers.
            if not has_group_by and not is_distinct:
                all_t_cols = schema_tables[raw_t]
                # For very wide tables (>30 columns), only add ID/key columns to keep the SQL short.
                if len(all_t_cols) > 30:
                    key_cols = [
                        (cname, is_t)
                        for cname, is_t in all_t_cols
                        if cname in ("id", "uuid")
                        or cname.endswith("_key")
                        or cname.endswith("_id")
                    ]
                    chosen_cols = key_cols[:4] if key_cols else all_t_cols[:6]
                else:
                    chosen_cols = all_t_cols
                for cname, is_t in chosen_cols:
                    table_key_exprs.append((f"{q}{raw_a}{q}.{q}{cname}{q}", is_t))
        elif (
            raw_t in cte_tables
            and (has_bare_star or "fdc.*" in select_body.lower())
            and not has_group_by
            and not is_distinct
        ):
            # If the query selects `*` from a helper `WITH` table (CTE), use the CTE's columns.
            for cname, is_t in cte_tables[raw_t]:
                if has_bare_star:
                    table_key_exprs.append((f"{q}{cname}{q}", is_t))
                else:
                    table_key_exprs.append((f"{q}{raw_a}{q}.{q}{cname}{q}", is_t))

    # If the query uses `GROUP BY ROLLUP(...)` together with a window function,
    # include the `GROUP BY` columns as tie-breakers so summary (`NULL`) rows sort consistently.
    if (
        re.search(r"\bROLLUP\b", after_from_masked, flags=re.IGNORECASE)
        and "OVER" in select_body.upper()
    ):
        gb_m = re.search(
            r"\bGROUP\s+BY\s+(.*?)(?=\s+\b(?:WITH\s+ROLLUP|HAVING|ORDER\s+BY|LIMIT|WINDOW)\b|\s*$)",
            after_from_masked,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if gb_m:
            gb_raw = after_from_raw[gb_m.start(1) : gb_m.end(1)].strip()
            rollup_m = re.match(
                r"^ROLLUP\s*\((.*)\)\s*$",
                gb_raw,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if rollup_m:
                gb_raw = rollup_m.group(1).strip()
            for g_col in _split_top_level_commas(gb_raw):
                g_col = g_col.strip()
                if g_col:
                    is_tg = _is_expr_text_col(
                        g_col, alias_to_table, schema_tables, pure_text_cols
                    )
                    table_key_exprs.append((g_col, is_tg))

    # Collect (a) regular columns in the `SELECT` list and (b) columns passed into window functions.
    items = _split_top_level_commas(select_body)
    non_window_exprs: List[Tuple[str, bool]] = []
    window_agg_args: List[Tuple[str, bool]] = []
    for item in items:
        for arg in _extract_window_func_args(item):
            is_t = _is_expr_text_col(
                arg, alias_to_table, schema_tables, pure_text_cols
            )
            window_agg_args.append((arg, is_t))
        if "OVER" not in item.upper() and ":=" not in item:
            base_expr = _strip_select_alias(item)
            if (
                base_expr
                and "*" not in base_expr
                and not re.search(r"^\s*\(", base_expr)
            ):
                is_t = _is_expr_text_col(
                    base_expr, alias_to_table, schema_tables, pure_text_cols
                )
                non_window_exprs.append((base_expr, is_t))

    candidate_ties = non_window_exprs + window_agg_args + table_key_exprs
    if not candidate_ties:
        return block_sql

    formatted_ties = [
        _format_tie_expr(e, dialect, is_t, "ASC") for e, is_t in candidate_ties
    ]

    def rewrite_over(s_body: str) -> str:
        """Append tie-breaker columns to `OVER (... ORDER BY ...)` clauses in `s_body`."""
        out = []
        i = 0
        while i < len(s_body):
            over_m = re.search(r"\bOVER\s*\(", s_body[i:], flags=re.IGNORECASE)
            if not over_m:
                out.append(s_body[i:])
                break
            open_p = i + over_m.end() - 1
            close_p = _find_matching_paren(s_body, open_p)
            if close_p == -1:
                out.append(s_body[i:])
                break
            prefix_before_over = s_body[i : i + over_m.start()]
            # Do NOT add tie-breakers to `RANK()`, `DENSE_RANK()`, `PERCENT_RANK()`,
            # or `CUME_DIST()`, because tied rows in those functions are supposed to
            # receive the exact same rank number (adding unique tie-breakers would
            # turn `DENSE_RANK()` into `ROW_NUMBER()`).
            is_shared_tie_rank = bool(
                re.search(
                    r"\b(?:DENSE_RANK|RANK|PERCENT_RANK|CUME_DIST)\s*\(\s*\)\s*$",
                    prefix_before_over,
                    flags=re.IGNORECASE,
                )
            )
            out.append(s_body[i : open_p + 1])
            over_inner = s_body[open_p + 1 : close_p]
            # Find the `ORDER BY ...` list inside `OVER (...)`, stopping before `ROWS`/`RANGE`/`GROUPS`.
            ob_m = re.search(
                r"\bORDER\s+BY\s+(.*?)(?=\s+\b(?:ROWS|RANGE|GROUPS)\b|$)",
                over_inner,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if ob_m and not is_shared_tie_rank:
                ob_keys = _split_top_level_commas(ob_m.group(1).strip())
                first_k_clean = (
                    _extract_key_dir(ob_keys[0])[0].strip('`"').lower()
                    if ob_keys
                    else ""
                )
                # Keep the query's primary sort key (or 3-part date sort keys ending in `_sort`),
                # then append the shared tie-breaker columns so both MySQL and PostgreSQL
                # break ties on the primary key in the exact same order.
                if first_k_clean.endswith("_sort") and len(ob_keys) >= 3:
                    kept_keys = ob_keys[:3]
                elif not candidate_ties:
                    kept_keys = ob_keys
                else:
                    kept_keys = ob_keys[:1]
                primary_formatted = []
                for k in kept_keys:
                    bk, dk = _extract_key_dir(k)
                    is_tk = _is_expr_text_col(
                        bk, alias_to_table, schema_tables, pure_text_cols
                    )
                    primary_formatted.append(
                        _format_tie_expr(bk, dialect, is_tk, dk)
                    )
                all_ob = primary_formatted + formatted_ties
                new_ob = "ORDER BY " + ", ".join(all_ob)
                over_inner = (
                    over_inner[: ob_m.start()]
                    + new_ob
                    + over_inner[ob_m.end() :]
                )
            out.append(over_inner + ")")
            i = close_p + 1
        return "".join(out)

    new_select_body = rewrite_over(select_body)

    # Also check if this `SELECT` block ends with an `ORDER BY ... LIMIT` top-N clause
    # (or a MySQL `@rank :=` user-variable ranking query) that needs tie-breakers.
    from_end_idx = from_m.end() + end_m.start() if end_m else len(block_sql)
    tail_segment = block_sql[from_end_idx:]
    tail_masked = masked[from_end_idx:]
    ob_tail_m = re.search(
        r"\bORDER\s+BY\s+(.*?)(?=\s+\b(?:LIMIT|OFFSET|FETCH|FOR)\b|\s*;?\s*$)",
        tail_masked,
        flags=re.IGNORECASE | re.DOTALL,
    )
    has_limit = bool(re.search(r"\bLIMIT\b", tail_masked, flags=re.IGNORECASE))
    has_user_var = ":=" in select_body
    is_star_cte_order = (
        dialect == "mysql"
        and has_bare_star
        and bool(table_key_exprs)
        and not has_limit
    )
    if ob_tail_m and (has_limit or has_user_var or is_star_cte_order):
        raw_ob_body = tail_segment[ob_tail_m.start(1) : ob_tail_m.end(1)].strip()
        ob_keys = _split_top_level_commas(raw_ob_body)
        kept_count = (
            2
            if (has_user_var or is_star_cte_order) and len(ob_keys) >= 2
            else 1
        )
        kept_keys = ob_keys[:kept_count]
        primary_formatted = []
        for k in kept_keys:
            bk, dk = _extract_key_dir(k)
            is_tk = _is_expr_text_col(
                bk, alias_to_table, schema_tables, pure_text_cols
            )
            primary_formatted.append(_format_tie_expr(bk, dialect, is_tk, dk))
        tail_ties = [
            _format_tie_expr(e, dialect, is_t, "ASC")
            for e, is_t in non_window_exprs
        ]
        if not has_group_by and not is_distinct:
            tail_ties.extend(
                _format_tie_expr(e, dialect, is_t, "ASC")
                for e, is_t in table_key_exprs
            )
        all_ob = primary_formatted + tail_ties
        new_ob_clause = "ORDER BY " + ", ".join(all_ob)
        if is_star_cte_order:
            new_ob_clause += " LIMIT 18446744073709551615"
        tail_segment = (
            tail_segment[: ob_tail_m.start()]
            + new_ob_clause
            + tail_segment[ob_tail_m.end() :]
        )

    return (
        block_sql[: select_m.end()]
        + new_select_body
        + block_sql[from_m.start() : from_end_idx]
        + tail_segment
    )


def _inject_deterministic_tiebreakers(
    sql: str,
    dialect: str = "",
    schema_ddl: str = "",
) -> str:
    """Add shared tie-breaker columns to `ORDER BY` clauses across all `SELECT` blocks in `sql`.

    When a query sorts rows that have ties (for example, inside `ROW_NUMBER()`,
    `LAG()`, `LEAD()`, a moving average over adjacent rows, or `ORDER BY ... LIMIT`),
    MySQL and PostgreSQL can visit the tied rows in different orders and produce
    different numbers. This helper appends the remaining columns of the row (and
    the table's ID/key columns) to the `ORDER BY` list in both queries so both
    databases break ties the exact same way. Functions where tied rows are
    supposed to share the same rank (`RANK()`, `DENSE_RANK()`) are left untouched.

    Args:
        sql: Original or converted SQL query string.
        dialect: SQL dialect (`"mysql"`, `"postgresql"`, or `""` for plain SQL).
        schema_ddl: Optional database table definition text (`CREATE TABLE ...`)
            so the helper knows (1) what extra columns exist on each table if
            the columns in `SELECT` aren't enough to break ties, and (2) which
            columns are text vs. numbers/dates (so case-insensitive `UPPER(...)`
            sorting is only applied to text columns).

    Returns:
        SQL query string with shared tie-breaker columns added to its `ORDER BY` clauses.
    """
    clean_sql = sql.strip().rstrip(";").strip()
    schema_tables, pure_text_cols = _parse_schema_columns(schema_ddl)
    cte_tables = _extract_cte_columns(clean_sql, pure_text_cols)

    def process_rec(text: str) -> str:
        """Walk through parenthesized subqueries from inside out, rewriting each `SELECT` block."""
        out = []
        i = 0
        while i < len(text):
            open_p = -1
            in_sq = False
            in_dq = False
            for j in range(i, len(text)):
                ch = text[j]
                if ch == "'" and not in_dq:
                    in_sq = not in_sq
                elif ch == '"' and not in_sq:
                    in_dq = not in_dq
                elif not in_sq and not in_dq and ch == "(":
                    open_p = j
                    break
            if open_p == -1:
                out.append(text[i:])
                break
            close_p = _find_matching_paren(text, open_p)
            if close_p == -1:
                out.append(text[i:])
                break
            prefix = text[i:open_p]
            inner = text[open_p + 1 : close_p]
            out.append(prefix + "(" + process_rec(inner) + ")")
            i = close_p + 1

        combined = "".join(out)
        masked = _mask_parens_and_strings(combined)
        # If this is a top-level `WITH ... SELECT ...` query, rewrite the final `SELECT` part
        # (the inner `WITH` subqueries were already rewritten inside the parentheses above).
        if re.match(r"^\s*WITH\b", masked, flags=re.IGNORECASE):
            sel_m = re.search(r"\bSELECT\b", masked, flags=re.IGNORECASE)
            if sel_m:
                return combined[: sel_m.start()] + _inject_tiebreakers_in_select(
                    combined[sel_m.start() :],
                    dialect,
                    schema_tables,
                    pure_text_cols,
                    cte_tables,
                )
            return combined
        if re.match(r"^\s*SELECT\b", masked, flags=re.IGNORECASE):
            return _inject_tiebreakers_in_select(
                combined, dialect, schema_tables, pure_text_cols, cte_tables
            )
        return combined

    return process_rec(clean_sql)


def _inject_window_tiebreakers(
    sql: str,
    dialect: str = "",
    schema_ddl: str = "",
) -> str:
    """Alias for `_inject_deterministic_tiebreakers` kept for unit-test compatibility.

    Args:
        sql: Original or converted SQL query string.
        dialect: Optional SQL dialect (`"mysql"`, `"postgresql"`, or `""`).
        schema_ddl: Optional database table definition text (`CREATE TABLE ...`)
            listing each table's column names and types (text vs. number/date).

    Returns:
        SQL query string with shared tie-breaker columns added to its `ORDER BY` clauses.
    """
    return _inject_deterministic_tiebreakers(
        sql, dialect=dialect, schema_ddl=schema_ddl
    )


class QueryConversionOrchestrator:
    """Manages query conversion with iterative refinement."""
    
    def __init__(
        self,
        converter: QueryConverter,
        source_connection: Dict,
        dest_connection: Dict,
        source_schema: str,
        dest_schema: str,
        max_attempts: int = 5,
        num_workers: int = 1
    ):
        """Initialize query conversion orchestrator.
        
        Args:
            converter: QueryConverter instance
            source_connection: Source database connection info
            dest_connection: Destination database connection info
            source_schema: Source database schema
            dest_schema: Destination database schema
            max_attempts: Maximum attempts per query (default: 3)
            num_workers: Number of parallel workers (default: 1)
        """
        self.converter = converter
        self.source_connection = source_connection
        self.dest_connection = dest_connection
        self.source_schema = source_schema
        self.dest_schema = dest_schema
        self.max_attempts = max_attempts
        self.num_workers = num_workers
        self.executor = QueryExecutor()
        self.print_lock = Lock()
    
    def convert_query(self, source_query: str, question_id: str = "", verbose: bool = True) -> ConversionResult:
        """Convert a single query with iterative refinement.
        
        Args:
            source_query: Query in source dialect
            question_id: Optional question ID for logging
            verbose: Whether to print progress messages
            
        Returns:
            ConversionResult with final status and converted query
        """
        def safe_print(*args, **kwargs):
            """Thread-safe printing."""
            if verbose:
                with self.print_lock:
                    print(*args, **kwargs)
        
        safe_print(f"\n{'=' * 80}")
        safe_print(f"Converting query (ID: {question_id}):")
        safe_print(f"{source_query[:100]}{'...' if len(source_query) > 100 else ''}")
        safe_print(f"{'=' * 80}")
        
        # Execute source query
        safe_print("\n1. Executing source query...")
        source_result = self._execute_source_query(source_query)
        
        if source_result.error:
            safe_print(f"✗ Source query execution failed: {source_result.error}")
            return ConversionResult(
                status=ConversionStatus.UNABLE_TO_MATCH,
                converted_query="",
                source_result=source_result,
                reason=f"Source query execution failed: {source_result.error}",
                attempts=0
            )
        
        safe_print(f"✓ Source query executed: {source_result.total_rows} rows")
        
        # Iterative conversion loop
        converted_query = None
        dest_result = None
        last_notes = ""  # Track last notes from LLM
        attempt_history = []  # Track all previous attempts
        
        for attempt in range(1, self.max_attempts + 1):
            safe_print(f"\n2. AI Conversion Attempt {attempt}/{self.max_attempts}...")
            
            # Build prompt with full history
            prompt = self.converter.get_conversion_prompt(
                source_schema=self.source_schema,
                dest_schema=self.dest_schema,
                source_query=source_query,
                converted_query=converted_query,
                source_result=source_result,
                dest_result=dest_result,
                attempt=attempt,
                max_attempts=self.max_attempts,
                attempt_history=attempt_history
            )
            
            # Call AI (ReAct style)
            try:
                response = gemini_inference(prompt, temperature=0.2, enforce_json=True)
                result_json = json.loads(response)
                
                # Check if AI has finished (result_matched or unable_to_match)
                conversion_finished = result_json.get('conversion_finished', '')
                notes = result_json.get('notes', '')
                last_notes = notes  # Track last notes for potential exhausted retry
                
                if conversion_finished:
                    safe_print(f"   AI Decision: {conversion_finished}")
                    safe_print(f"   Notes: {notes}")
                    
                    # Parse the final status
                    try:
                        status = ConversionStatus(conversion_finished)
                    except ValueError:
                        safe_print(f"✗ Invalid conversion_finished value: {conversion_finished}")
                        continue
                    
                    # If result_matched but no converted_query provided, use source query
                    # (LLM is saying the query doesn't need conversion)
                    final_query = converted_query if converted_query else source_query
                    
                    # Return the final result
                    return ConversionResult(
                        status=status,
                        converted_query=final_query,
                        source_result=source_result,
                        dest_result=dest_result,
                        reason=notes,
                        attempts=attempt
                    )
                
                # AI wants to try a new conversion
                new_query = result_json.get('converted_query', '')
                if new_query:
                    converted_query = new_query
                    safe_print(f"   AI Action: Trying new conversion")
                    safe_print(f"   Notes: {notes}")
                    
                    # Execute the new converted query
                    safe_print(f"\n3. Executing converted query...")
                    dest_result = self._execute_dest_query(converted_query)
                    
                    if dest_result.error:
                        safe_print(f"✗ Execution failed: {dest_result.error}")
                    else:
                        safe_print(f"✓ Executed: {dest_result.total_rows} rows")
                        
                        # Only run tie-breaker checks if the converter explicitly
                        # enables allow_tie_equivalence (True on MySQLToPGQueryConverter;
                        # False on all other converters).
                        allow_tie_eq = getattr(
                            self.converter, 'allow_tie_equivalence', False
                        )
                        matches, match_reason = self.converter.compare_results(
                            source_result, dest_result
                        )
                        if (
                            not matches
                            and allow_tie_eq
                            and len(source_result.columns) == len(dest_result.columns)
                        ):
                            tb_matches, tb_reason = self._check_shared_window_tiebreaker(
                                source_query, converted_query
                            )
                            if tb_matches:
                                matches, match_reason = True, tb_reason
                        if matches:
                            safe_print(f"✓ Results MATCH! {match_reason}")
                            return ConversionResult(
                                status=ConversionStatus.RESULT_MATCHED,
                                converted_query=converted_query,
                                source_result=source_result,
                                dest_result=dest_result,
                                reason=match_reason,
                                attempts=attempt
                            )
                        else:
                            safe_print(f"✗ Results differ: {match_reason}")
                    
                    # Record this attempt in history
                    attempt_history.append({
                        'attempt': attempt,
                        'query': converted_query,
                        'result': dest_result,
                        'notes': notes
                    })
                    
                    # Continue to next attempt with updated dest_result
                    continue
                else:
                    safe_print(f"✗ AI response missing both 'conversion_finished' and 'converted_query'")
                    continue
                
            except Exception as e:
                safe_print(f"✗ AI conversion failed: {e}")
                continue
        
        # Max attempts reached without success
        return ConversionResult(
            status=ConversionStatus.EXHAUSTED_RETRY,
            converted_query=converted_query or "",
            source_result=source_result,
            dest_result=dest_result,
            reason=last_notes or f"Max attempts ({self.max_attempts}) reached without achieving match",
            attempts=self.max_attempts
        )
    
    def convert_queries(
        self,
        queries: List[Tuple[str, str]],  # List of (question_id, query)
        output_file: Optional[Path] = None
    ) -> Dict[str, ConversionResult]:
        """Convert multiple queries.
        
        Args:
            queries: List of tuples (question_id, query)
            output_file: Optional file to save conversion results
            
        Returns:
            Dict mapping (question_id, original query) to ConversionResult
        """
        results = {}
        
        print(f"\n{'=' * 80}")
        print(f"QUERY CONVERSION: Converting {len(queries)} queries")
        if self.num_workers > 1:
            print(f"Using {self.num_workers} parallel workers")
        print(f"{'=' * 80}")
        
        if self.num_workers == 1:
            # Sequential processing
            for i, (question_id, query) in enumerate(queries, 1):
                print(f"\n\n{'#' * 80}")
                print(f"QUERY {i}/{len(queries)} (Question ID: {question_id})")
                print(f"{'#' * 80}")
                
                result = self.convert_query(query, question_id=question_id, verbose=True)
                results[(question_id, query)] = result
                
                # Print summary
                print(f"\n{'=' * 80}")
                print(f"RESULT: {result.status.value}")
                print(f"Attempts: {result.attempts}")
                if result.reason:
                    print(f"Reason: {result.reason}")
                if result.status == ConversionStatus.RESULT_MATCHED:
                    print(f"✓ SUCCESS: Query converted and verified!")
                elif result.status == ConversionStatus.EXHAUSTED_RETRY:
                    print(f"⚠ WARNING: Max attempts reached without match")
                else:
                    print(f"✗ ERROR: Unable to match query")
                print(f"{'=' * 80}")
        else:
            # Parallel processing
            completed = 0
            total = len(queries)
            
            def convert_with_progress(question_id: str, query: str, index: int) -> Tuple[str, str, ConversionResult]:
                """Convert query and return with identifiers."""
                result = self.convert_query(query, question_id=question_id, verbose=False)
                return (question_id, query, result)
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                future_to_query = {
                    executor.submit(convert_with_progress, qid, q, i): (qid, q, i)
                    for i, (qid, q) in enumerate(queries, 1)
                }
                
                # Process completed tasks
                for future in as_completed(future_to_query):
                    qid, query, result = future.result()
                    results[(qid, query)] = result
                    completed += 1
                    
                    # Print progress
                    with self.print_lock:
                        if result.status == ConversionStatus.RESULT_MATCHED:
                            status_symbol = "✓"
                        elif result.status == ConversionStatus.EXHAUSTED_RETRY:
                            status_symbol = "⚠"
                        else:
                            status_symbol = "✗"
                        print(f"[{completed:2d}/{total}] {status_symbol} Query {qid}: {result.status.value}")
        
        # Save results if requested
        if output_file:
            self._save_results_with_ids(results, output_file)
        
        # Print final summary
        self._print_summary(results)
        
        return results
    
    @staticmethod
    def load_queries_from_csv(csv_file: Path) -> List[Tuple[str, str]]:
        """Load queries from CSV file.
        
        Args:
            csv_file: Path to CSV file with columns: question_id, source_query
            
        Returns:
            List of tuples (question_id, query)
        """
        queries = []
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question_id = row.get('question_id', '')
                query = row.get('source_query', '')
                if query:
                    queries.append((question_id, query))
        return queries
    
    def _execute_source_query(self, query: str) -> QueryResult:
        """Execute query on source database."""
        if self.converter.source_dialect == 'sqlite':
            return self.executor.execute_sqlite(
                self.source_connection['path'],
                query
            )
        elif self.converter.source_dialect == 'postgresql':
            return self.executor.execute_postgresql(
                self.source_connection['host'],
                self.source_connection['port'],
                self.source_connection['user'],
                self.source_connection['password'],
                self.source_connection['database'],
                query
            )
        elif self.converter.source_dialect == 'mysql':
            return self.executor.execute_mysql(
                self.source_connection['host'],
                self.source_connection['port'],
                self.source_connection['user'],
                self.source_connection['password'],
                self.source_connection['database'],
                query
            )
        elif self.converter.source_dialect == 'bigquery':
            return self.executor.execute_bigquery(
                self.source_connection['project_id'],
                self.source_connection['dataset_id'],
                query
            )
        else:
            raise NotImplementedError(f"Source dialect {self.converter.source_dialect} not supported yet")

    def _check_shared_window_tiebreaker(
        self,
        source_query: str,
        converted_query: str,
    ) -> Tuple[bool, str]:
        """Re-run both queries with the same tie-breaker columns and check if 100% of columns match.

        When a query has ties in an `ORDER BY` clause (for example inside
        `ROW_NUMBER()`, `LAG()`, `LEAD()`, a moving average over adjacent rows,
        or `ORDER BY ... LIMIT`), MySQL and PostgreSQL may break those ties in
        different orders and produce different numbers. This method temporarily
        adds the remaining columns of the row (and the table's ID/key columns)
        as shared tie-breakers to both `source_query` and `converted_query`,
        re-runs both, and checks `compare_results` across 100% of columns.

        Args:
            source_query: Original SQL query in the source dialect (MySQL).
            converted_query: Candidate SQL query in the destination dialect (PostgreSQL).

        Returns:
            Tuple of `(matches, reason)` after running both queries with shared tie-breakers.
        """
        schema_ddl = self.dest_schema or self.source_schema or ""
        src_tb = _inject_deterministic_tiebreakers(
            source_query,
            dialect=self.converter.source_dialect,
            schema_ddl=schema_ddl,
        )
        dest_tb = _inject_deterministic_tiebreakers(
            converted_query,
            dialect=self.converter.dest_dialect,
            schema_ddl=schema_ddl,
        )
        # If neither query had any `OVER (... ORDER BY ...)` or `ORDER BY ... LIMIT`
        # clause to add tie-breakers to, there is nothing to re-run.
        if src_tb == source_query and dest_tb == converted_query:
            return False, "No deterministic tie-breakers injected"
        r_src = self._execute_source_query(src_tb)
        r_dest = self._execute_dest_query(dest_tb)
        if r_src.error or r_dest.error:
            return False, "Tie-broken query execution failed"
        ok, reason = self.converter.compare_results(r_src, r_dest)
        if ok:
            return (
                True,
                f"All checks passed (shared window tie-breaker: {r_src.total_rows} rows): {reason}",
            )
        return False, reason
    
    def _execute_dest_query(self, query: str) -> QueryResult:
        """Execute query on destination database."""
        if self.converter.dest_dialect == 'postgresql':
            return self.executor.execute_postgresql(
                self.dest_connection['host'],
                self.dest_connection['port'],
                self.dest_connection['user'],
                self.dest_connection['password'],
                self.dest_connection['database'],
                query
            )
        elif self.converter.dest_dialect == 'mysql':
            return self.executor.execute_mysql(
                self.dest_connection['host'],
                self.dest_connection['port'],
                self.dest_connection['user'],
                self.dest_connection['password'],
                self.dest_connection['database'],
                query
            )
        elif self.converter.dest_dialect == 'spanner':
            return self.executor.execute_spanner(
                self.dest_connection['project_id'],
                self.dest_connection['instance_id'],
                self.dest_connection['database_id'],
                query
            )
        elif self.converter.dest_dialect == 'bigquery':
            return self.executor.execute_bigquery(
                self.dest_connection['project_id'],
                self.dest_connection['dataset_id'],
                query
            )
        else:
            raise NotImplementedError(f"Destination dialect {self.converter.dest_dialect} not supported yet")
    
    def _save_results_with_ids(self, results: Dict[Tuple[str, str], ConversionResult], output_file: Path):
        """Save conversion results to file (CSV or JSON) with question IDs."""
        if str(output_file).endswith('.csv'):
            self._save_results_csv_with_ids(results, output_file)
        else:
            self._save_results_json_with_ids(results, output_file)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    def _save_results_csv_with_ids(self, results: Dict[Tuple[str, str], ConversionResult], output_file: Path):
        """Save conversion results to CSV file with question IDs."""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'question_id',
                'source_query',
                'converted_query',
                'conversion_result',
                'attempts',
                'source_rows',
                'dest_rows',
                'reason'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for (question_id, query), result in results.items():
                writer.writerow({
                    'question_id': question_id,
                    'source_query': query,
                    'converted_query': result.converted_query,
                    'conversion_result': result.status.value,
                    'attempts': result.attempts,
                    'source_rows': result.source_result.total_rows if result.source_result else 0,
                    'dest_rows': result.dest_result.total_rows if result.dest_result else 0,
                    'reason': result.reason or ''
                })
    
    def _save_results_json_with_ids(self, results: Dict[Tuple[str, str], ConversionResult], output_file: Path):
        """Save conversion results to JSON file with question IDs."""
        output_data = []
        
        for (question_id, query), result in results.items():
            output_data.append({
                'question_id': question_id,
                'source_query': query,
                'converted_query': result.converted_query,
                'status': result.status.value,
                'reason': result.reason,
                'attempts': result.attempts,
                'source_rows': result.source_result.total_rows if result.source_result else 0,
                'dest_rows': result.dest_result.total_rows if result.dest_result else 0
            })
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def _print_summary(self, results: Dict):
        """Print summary of all conversions."""
        total = len(results)
        matched = sum(1 for r in results.values() if r.status == ConversionStatus.RESULT_MATCHED)
        unable = sum(1 for r in results.values() if r.status == ConversionStatus.UNABLE_TO_MATCH)
        exhausted = sum(1 for r in results.values() if r.status == ConversionStatus.EXHAUSTED_RETRY)
        
        print(f"\n\n{'=' * 80}")
        print(f"FINAL SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total queries: {total}")
        print(f"✓ Result Matched: {matched} ({matched/total*100:.1f}%)")
        print(f"✗ Unable to Match: {unable} ({unable/total*100:.1f}%)")
        print(f"⚠ Exhausted Retry: {exhausted} ({exhausted/total*100:.1f}%)")
        print(f"{'=' * 80}\n")
