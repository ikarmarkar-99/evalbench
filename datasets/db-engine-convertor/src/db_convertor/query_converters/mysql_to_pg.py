"""MySQL to PostgreSQL SQL query converter with tie-aware mechanical checking.

This module contains:
  - SQL provenance helpers (`_ColumnProvenance`, `_strip_sql_comments`,
    `_find_matching_paren`, `_split_top_level_commas`,
    `_split_top_level_keyword`, `_analyze_query_columns`, and
    `_identify_tie_sensitive_columns`) that determine which output `SELECT`
    columns genuinely derive from tie-sensitive physical-order window or top-1
    constructs.
  - `MySQLToPGQueryConverter`: Builds the ReAct prompt for converting MySQL
    queries to PostgreSQL and implements `compare_results` with tightened
    mechanical tie-equivalence checks.
"""

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from .base import QueryConverter, QueryResult


# Window functions whose output depends on the physical ordering of tied rows.
# Peer-invariant functions such as RANK(), DENSE_RANK(), PERCENT_RANK(), and
# CUME_DIST() assign identical values to all tied peers and are excluded.
_PHYSICAL_ORDER_WINDOW_FUNCS_RE = re.compile(
    r"\b(?:ROW_NUMBER|NTILE|LAG|LEAD|FIRST_VALUE|LAST_VALUE|NTH_VALUE)\s*\(",
    re.IGNORECASE,
)

# SQL keywords that cannot be table or CTE aliases in FROM / JOIN clauses.
_SQL_CLAUSE_KEYWORDS = frozenset({
    "on",
    "where",
    "group",
    "order",
    "limit",
    "offset",
    "fetch",
    "join",
    "left",
    "right",
    "inner",
    "outer",
    "cross",
    "full",
    "natural",
    "using",
    "having",
    "union",
    "intersect",
    "except",
    "window",
    "as",
    "with",
})

# Negative-lookahead pattern preventing FROM/JOIN alias regexes from consuming
# an adjacent SQL keyword when a table has no alias.
_ALIAS_KEYWORD_LOOKAHEAD = (
    r"(?!\b(?:"
    + "|".join(sorted(_SQL_CLAUSE_KEYWORDS))
    + r")\b)"
)


@dataclass(frozen=True)
class _ColumnProvenance:
    """Provenance metadata for a single projected SQL output column."""

    name: str
    is_tie_sensitive: bool
    raw_expr: str
    provenance_sql: str
    is_outer_unfiltered_rownum: bool = False
    allows_null_shift: bool = False


def _strip_sql_comments(sql: str) -> str:
    """Remove `--` line comments and `/* ... */` block comments outside quotes.

    Args:
        sql: Raw SQL query string.

    Returns:
        SQL query string with comments replaced by whitespace.
    """
    out: List[str] = []
    i = 0
    n = len(sql)
    in_sq = False
    in_dq = False
    in_bt = False
    while i < n:
        ch = sql[i]
        if ch == "'" and not in_dq and not in_bt:
            in_sq = not in_sq
            out.append(ch)
            i += 1
        elif ch == '"' and not in_sq and not in_bt:
            in_dq = not in_dq
            out.append(ch)
            i += 1
        elif ch == "`" and not in_sq and not in_dq:
            in_bt = not in_bt
            out.append(ch)
            i += 1
        elif not in_sq and not in_dq and not in_bt:
            if sql[i : i + 2] == "--":
                nl = sql.find("\n", i + 2)
                if nl == -1:
                    break
                out.append("\n")
                i = nl + 1
                continue
            if sql[i : i + 2] == "/*":
                end = sql.find("*/", i + 2)
                if end == -1:
                    break
                out.append(" ")
                i = end + 2
                continue
            out.append(ch)
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _find_matching_paren(sql: str, open_idx: int) -> int:
    """Find the closing `)` matching the `(` at `open_idx`, ignoring quoted text.

    Args:
        sql: SQL text containing the parenthesized block.
        open_idx: Zero-based index of the opening `(`.

    Returns:
        Zero-based index of the matching `)`, or `-1` if unbalanced.
    """
    depth = 0
    in_sq = False
    in_dq = False
    in_bt = False
    for i in range(open_idx, len(sql)):
        ch = sql[i]
        if ch == "'" and not in_dq and not in_bt:
            in_sq = not in_sq
        elif ch == '"' and not in_sq and not in_bt:
            in_dq = not in_dq
        elif ch == "`" and not in_sq and not in_dq:
            in_bt = not in_bt
        elif not in_sq and not in_dq and not in_bt:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
    return -1


def _build_top_level_mask(sql: str, preserve_quotes: bool = False) -> str:
    """Mask parenthesized subexpressions and quoted literals to spaces.

    Args:
        sql: SQL expression or clause string.
        preserve_quotes: If True, replaces characters inside top-level quoted
            identifiers with placeholder `'a'` while preserving their quote
            delimiters so trailing `AS "alias"` patterns remain matchable.

    Returns:
        Same-length string with nested parentheses masked out.
    """
    mask = [" "] * len(sql)
    depth = 0
    in_sq = False
    in_dq = False
    in_bt = False
    for i, ch in enumerate(sql):
        if ch == "'" and not in_dq and not in_bt:
            in_sq = not in_sq
        elif ch == '"' and not in_sq and not in_bt:
            in_dq = not in_dq
            if preserve_quotes and depth == 0:
                mask[i] = '"'
        elif ch == "`" and not in_sq and not in_dq:
            in_bt = not in_bt
            if preserve_quotes and depth == 0:
                mask[i] = "`"
        elif not in_sq and not in_dq and not in_bt:
            if ch == "(":
                if depth == 0:
                    mask[i] = "("
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
                if depth == 0:
                    mask[i] = ")"
            elif depth == 0:
                mask[i] = ch
        elif preserve_quotes and depth == 0 and (in_dq or in_bt):
            mask[i] = "a"
    return "".join(mask)


def _split_top_level_commas(text: str) -> List[str]:
    """Split a SQL expression list on commas at parenthesis depth zero.

    Args:
        text: Comma-separated SQL projection or expression list.

    Returns:
        List of trimmed top-level SQL expressions.
    """
    parts: List[str] = []
    cur: List[str] = []
    depth = 0
    in_sq = False
    in_dq = False
    in_bt = False
    for ch in text:
        if ch == "'" and not in_dq and not in_bt:
            in_sq = not in_sq
        elif ch == '"' and not in_sq and not in_bt:
            in_dq = not in_dq
        elif ch == "`" and not in_sq and not in_dq:
            in_bt = not in_bt
        elif not in_sq and not in_dq and not in_bt:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                parts.append("".join(cur).strip())
                cur = []
                continue
        cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def _split_top_level_keyword(sql: str, keyword_pattern: str) -> List[str]:
    """Split `sql` at top-level occurrences of `keyword_pattern`.

    Args:
        sql: SQL query or clause string.
        keyword_pattern: Regular expression matching the separator keyword.

    Returns:
        Non-empty trimmed segments separated by top-level matches.
    """
    top_str = _build_top_level_mask(sql, preserve_quotes=False)
    spans = [
        m.span() for m in re.finditer(keyword_pattern, top_str, flags=re.IGNORECASE)
    ]
    if not spans:
        return [sql.strip()]
    parts: List[str] = []
    prev = 0
    for start, end in spans:
        parts.append(sql[prev:start].strip())
        prev = end
    parts.append(sql[prev:].strip())
    return [p for p in parts if p]


def _unquote_ident(ident: str) -> str:
    """Strip surrounding double quotes or backticks and lowercase an identifier.

    Args:
        ident: Raw SQL identifier token.

    Returns:
        Normalized lowercase identifier name.
    """
    ident = ident.strip()
    if len(ident) >= 2 and ident[0] in ('"', "`") and ident[-1] == ident[0]:
        ident = ident[1:-1]
    return ident.lower()


def _extract_output_col_name(expr: str) -> str:
    """Extract the output column alias or bare column name from a SELECT item.

    Args:
        expr: Single `SELECT` projection item.

    Returns:
        Lowercase output column name, or empty string for unnamed expressions.
    """
    top = _build_top_level_mask(expr, preserve_quotes=True)
    # Match a top-level trailing `AS <alias>` (quoted or unquoted).
    as_matches = list(
        re.finditer(
            r"\bAS\s+(`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)\s*$",
            top,
            flags=re.IGNORECASE,
        )
    )
    if as_matches:
        start, end = as_matches[-1].span(1)
        return _unquote_ident(expr[start:end])

    # Match a bare `table.col` or `col` reference with no function calls.
    tail_match = re.search(
        r"(?:`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)\s*$", expr.strip()
    )
    if tail_match and "(" not in expr:
        return _unquote_ident(tail_match.group(0))
    return ""


def _strip_outer_alias(expr: str) -> str:
    """Remove a top-level trailing `AS <alias>` clause from a SELECT expression.

    Args:
        expr: Single `SELECT` projection item.

    Returns:
        Expression body without its trailing `AS <alias>`.
    """
    top = _build_top_level_mask(expr, preserve_quotes=True)
    alias_match = re.search(
        r"\s+\bAS\s+(?:`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)\s*$",
        top,
        flags=re.IGNORECASE,
    )
    if alias_match:
        return expr[: alias_match.start()].strip()
    return expr.strip()


def _expr_has_tie_sensitive_window(expr: str) -> Tuple[bool, bool]:
    """Check whether `expr` directly computes a tie-sensitive window or top-1 label.

    Only physical-order constructs can produce different values on tied sort keys:
      1. `ROW_NUMBER`, `NTILE`, `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`, or
         `NTH_VALUE` with an `ORDER BY` inside `OVER (...)`.
      2. Sliding/cumulative row frames `OVER (... ORDER BY ... ROWS ...)` on
         non-ranking functions.
      3. MySQL user-variable ranking (`@rank := ...`).
      4. Scalar top-1 label subqueries `(SELECT col FROM ... ORDER BY ... LIMIT 1)`
         that do not compute aggregates or arithmetic division.

    Peer-invariant constructs (`RANK`, `DENSE_RANK`, unframed `SUM/AVG OVER (ORDER BY ...)`
    which default to `RANGE`, and unordered `OVER (PARTITION BY ...)`) are excluded.

    Args:
        expr: SQL SELECT expression with outer alias stripped.

    Returns:
        Tuple `(is_tie_sensitive, allows_null_shift)` where `allows_null_shift`
        is True when tie reordering can change NULL count (e.g. `LAG`/`LEAD` or
        scalar `LIMIT 1` subqueries).
    """
    # Check for MySQL `@rank :=` user-variable ranking.
    if re.search(r"@[A-Za-z0-9_]*rank[A-Za-z0-9_]*\s*:=", expr, re.IGNORECASE):
        return True, False

    # Check for scalar top-1 label subqueries `(SELECT <label> ... ORDER BY ... LIMIT 1)`.
    stripped = expr.strip()
    if (
        stripped.startswith("(")
        and stripped.endswith(")")
        and _find_matching_paren(stripped, 0) == len(stripped) - 1
    ):
        inner = stripped[1:-1].strip()
        if re.match(r"^SELECT\b", inner, re.IGNORECASE) and re.search(
            r"\bORDER\s+BY\b.*\bLIMIT\s+1\s*$",
            inner,
            re.IGNORECASE | re.DOTALL,
        ):
            from_parts = _split_top_level_keyword(inner, r"\bFROM\b")
            if len(from_parts) >= 2:
                # Strip leading SELECT [DISTINCT] to inspect the projected scalar value.
                proj = re.sub(
                    r"^SELECT\s+(?:DISTINCT\s+)?",
                    "",
                    from_parts[0],
                    flags=re.IGNORECASE,
                ).strip()
                # Never allow aggregate or division scalar subqueries to be treated as tie labels.
                has_agg_or_div = re.search(
                    r"\b(?:AVG|SUM|COUNT|MIN|MAX|STDDEV|VARIANCE|VAR_POP|STDDEV_POP)\s*\(|/",
                    proj,
                    re.IGNORECASE,
                )
                if not has_agg_or_div:
                    return True, True

    # Scan every `OVER (...)` window specification in the expression.
    is_ts = False
    null_shift = False
    for over_match in re.finditer(r"\bOVER\s*\(", expr, flags=re.IGNORECASE):
        open_idx = over_match.end() - 1
        close_idx = _find_matching_paren(expr, open_idx)
        if close_idx == -1:
            continue
        over_body = expr[open_idx + 1 : close_idx]
        # Without ORDER BY inside OVER (...), window results are set-deterministic.
        if not re.search(r"\bORDER\s+BY\b", over_body, re.IGNORECASE):
            continue

        prefix = expr[: over_match.start()]
        # Exclude peer-group-invariant ranking functions (RANK, DENSE_RANK, etc.).
        if re.search(
            r"\b(?:DENSE_RANK|RANK|PERCENT_RANK|CUME_DIST)\s*\(\s*\)\s*$",
            prefix,
            re.IGNORECASE,
        ):
            continue

        if re.search(
            r"\b(?:LAG|LEAD|FIRST_VALUE|LAST_VALUE|NTH_VALUE)\s*\(",
            prefix,
            re.IGNORECASE,
        ):
            is_ts = True
            null_shift = True
        elif _PHYSICAL_ORDER_WINDOW_FUNCS_RE.search(prefix):
            is_ts = True
        elif re.search(r"\bROWS\b", over_body, re.IGNORECASE):
            is_ts = True

    return is_ts, null_shift


def _expr_is_direct_unfiltered_rownum(expr: str) -> bool:
    """Return True if `expr` is a direct `ROW_NUMBER() OVER (...)` or `NTILE(...) OVER (...)`.

    When `ROW_NUMBER()` or `NTILE()` appears directly in the outer `SELECT` list,
    it runs after all `FROM`, `JOIN`, and `WHERE` filtering, so every partition
    always emits the exact same multiset of rank integers regardless of ties.

    Args:
        expr: Raw outer `SELECT` item.

    Returns:
        True if `expr` is a bare `ROW_NUMBER` or `NTILE` window call.
    """
    body = _strip_outer_alias(expr).strip()
    if not re.match(
        r"^(?:ROW_NUMBER|NTILE)\s*\([^()]*\)\s*OVER\s*\(", body, re.IGNORECASE
    ):
        return False
    over_pos = body.upper().find("OVER")
    paren_pos = body.find("(", over_pos)
    return _find_matching_paren(body, paren_pos) == len(body) - 1


def _split_with_and_main_query(sql: str) -> Tuple[List[Tuple[str, str]], str]:
    """Split an optional `WITH cte AS (...), ...` prefix from the main SELECT.

    Supports `WITH RECURSIVE` and PostgreSQL `AS MATERIALIZED` / `AS NOT MATERIALIZED`.

    Args:
        sql: Cleaned SQL query string without comments.

    Returns:
        Tuple of `(cte_list, main_query_sql)` where `cte_list` contains
        `(lowercase_cte_name, cte_query_sql)` pairs in declaration order.
    """
    s = sql.strip().rstrip(";").strip()
    while s.startswith("(") and _find_matching_paren(s, 0) == len(s) - 1:
        s = s[1:-1].strip()

    with_match = re.match(r"^WITH\s+(?:RECURSIVE\s+)?", s, flags=re.IGNORECASE)
    if not with_match:
        return [], s

    idx = with_match.end()
    ctes: List[Tuple[str, str]] = []
    n = len(s)
    # Match `<cte_name> AS [NOT] [MATERIALIZED] ( ... )`.
    cte_header_re = re.compile(
        r"\s*(`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)\s+AS\s+"
        r"(?:(?:NOT\s+)?MATERIALIZED\s+)?\(",
        re.IGNORECASE,
    )
    while idx < n:
        header_match = cte_header_re.match(s[idx:])
        if not header_match:
            break
        cte_name = _unquote_ident(header_match.group(1))
        open_paren = idx + header_match.end() - 1
        close_paren = _find_matching_paren(s, open_paren)
        if close_paren == -1:
            break
        cte_body = s[open_paren + 1 : close_paren].strip()
        ctes.append((cte_name, cte_body))
        idx = close_paren + 1
        comma_match = re.match(r"\s*,\s*", s[idx:])
        if comma_match:
            idx += comma_match.end()
        else:
            break
    return ctes, s[idx:].strip()


def _extract_select_items_and_from(select_sql: str) -> Tuple[List[str], str]:
    """Split a single `SELECT ... FROM ...` branch into SELECT items and FROM tail.

    Args:
        select_sql: Single `SELECT` query branch (without `WITH` or `UNION`).

    Returns:
        Tuple of `(select_expressions, from_and_following_clauses)`.
    """
    s = select_sql.strip()
    while s.startswith("(") and _find_matching_paren(s, 0) == len(s) - 1:
        s = s[1:-1].strip()
    top_str = _build_top_level_mask(s, preserve_quotes=False)
    from_match = re.search(r"\bFROM\b", top_str, flags=re.IGNORECASE)
    if from_match:
        head = s[: from_match.start()].strip()
        rest = s[from_match.start() :].strip()
    else:
        head = s
        rest = ""
    proj = re.sub(
        r"^SELECT\s+(?:DISTINCT\s+)?", "", head, flags=re.IGNORECASE
    ).strip()
    return _split_top_level_commas(proj), rest


def _extract_derived_subqueries_and_aliases(
    from_clause: str,
    known_ctes: Dict[str, List[_ColumnProvenance]],
) -> Dict[str, List[_ColumnProvenance]]:
    """Resolve derived tables, CTE aliases, and top-1 `rn = 1` joins in `from_clause`.

    Args:
        from_clause: `FROM ...` tail of a `SELECT` branch.
        known_ctes: Mapping of enclosing CTE names to their analyzed columns.

    Returns:
        Mapping from lowercase table/CTE alias to its list of `_ColumnProvenance`.
    """
    alias_to_cols: Dict[str, List[_ColumnProvenance]] = dict(known_ctes)

    # 1. Analyze derived tables `(SELECT ...) [AS] <alias>` in the FROM/JOIN clause.
    alias_after_paren_re = re.compile(
        r"\s*(?:AS\s+)?"
        + _ALIAS_KEYWORD_LOOKAHEAD
        + r"(`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)",
        re.IGNORECASE,
    )
    i = 0
    n = len(from_clause)
    in_sq = False
    in_dq = False
    in_bt = False
    while i < n:
        ch = from_clause[i]
        if ch == "'" and not in_dq and not in_bt:
            in_sq = not in_sq
        elif ch == '"' and not in_sq and not in_bt:
            in_dq = not in_dq
        elif ch == "`" and not in_sq and not in_dq:
            in_bt = not in_bt
        elif not in_sq and not in_dq and not in_bt and ch == "(":
            close_idx = _find_matching_paren(from_clause, i)
            if close_idx != -1:
                inner = from_clause[i + 1 : close_idx].strip()
                if re.match(r"^(?:WITH|SELECT)\b", inner, re.IGNORECASE):
                    after = from_clause[close_idx + 1 :]
                    alias_match = alias_after_paren_re.match(after)
                    if alias_match:
                        alias = _unquote_ident(alias_match.group(1))
                        if alias not in _SQL_CLAUSE_KEYWORDS:
                            alias_to_cols[alias] = _analyze_query_columns(
                                inner, known_ctes, is_outer=False
                            )
                i = close_idx + 1
                continue
        i += 1

    # 2. Bind `FROM/JOIN <cte_name> [AS] <alias>` outside parentheses.
    top_from = _build_top_level_mask(from_clause, preserve_quotes=False)
    # Use the original characters at top level (including quoted identifiers) by
    # blanking out only parenthesized regions `( ... )`.
    unparen_chars = list(from_clause)
    p_idx = 0
    in_sq = in_dq = in_bt = False
    while p_idx < n:
        ch = from_clause[p_idx]
        if ch == "'" and not in_dq and not in_bt:
            in_sq = not in_sq
        elif ch == '"' and not in_sq and not in_bt:
            in_dq = not in_dq
        elif ch == "`" and not in_sq and not in_dq:
            in_bt = not in_bt
        elif not in_sq and not in_dq and not in_bt and ch == "(":
            c_idx = _find_matching_paren(from_clause, p_idx)
            if c_idx != -1:
                for k in range(p_idx, c_idx + 1):
                    unparen_chars[k] = " "
                p_idx = c_idx + 1
                continue
        p_idx += 1
    top_from = "".join(unparen_chars)

    from_join_re = re.compile(
        r"\b(?:FROM|JOIN)\s+(`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)"
        r"(?:\s+(?:AS\s+)?"
        + _ALIAS_KEYWORD_LOOKAHEAD
        + r"(`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*))?",
        re.IGNORECASE,
    )
    for m in from_join_re.finditer(top_from):
        tbl = _unquote_ident(m.group(1))
        als = _unquote_ident(m.group(2)) if m.group(2) else ""
        if als in _SQL_CLAUSE_KEYWORDS:
            als = ""
        if tbl in alias_to_cols and als:
            alias_to_cols[als] = alias_to_cols[tbl]

    # 3. Detect top-1 filters `alias.rn = 1` / `WHERE rn = 1` on a ROW_NUMBER() column.
    # For a top-1 tie-broken subquery, non-partition, non-sort-key, non-aggregate
    # payload labels can differ on a tie, whereas the ORDER BY sort key and
    # aggregates MUST match and stay protected.
    rank1_aliases: Set[str] = set()
    rank1_re = re.compile(
        r"(?:(`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*)?"
        r"(`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)\s*=\s*1\b",
        re.IGNORECASE,
    )
    for m in rank1_re.finditer(top_from):
        qual = _unquote_ident(m.group(1)) if m.group(1) else ""
        rcol = _unquote_ident(m.group(2))
        targets = (
            [qual]
            if qual and qual in alias_to_cols
            else list(alias_to_cols.keys())
        )
        for target in targets:
            for col_prov in alias_to_cols.get(target, []):
                if col_prov.name == rcol and col_prov.is_tie_sensitive and (
                    re.search(
                        r"\bROW_NUMBER\s*\(",
                        col_prov.provenance_sql,
                        re.IGNORECASE,
                    )
                    or re.search(
                        r"@[A-Za-z0-9_]*rank[A-Za-z0-9_]*\s*:=",
                        col_prov.provenance_sql,
                        re.IGNORECASE,
                    )
                ):
                    rank1_aliases.add(target)

    for target in rank1_aliases:
        cols_info = alias_to_cols.get(target, [])
        protected_cols: Set[str] = set()
        for col_prov in cols_info:
            # Never allow aggregate columns (such as COUNT/AVG/SUM sort keys) to be dropped.
            if re.search(
                r"\b(?:COUNT|SUM|AVG|MIN|MAX|STDDEV|VARIANCE|VAR_POP|STDDEV_POP)\s*\(",
                col_prov.provenance_sql,
                re.IGNORECASE,
            ):
                protected_cols.add(col_prov.name)
            if re.search(
                r"\bROW_NUMBER\s*\(", col_prov.provenance_sql, re.IGNORECASE
            ):
                over_match = re.search(
                    r"\bOVER\s*\((.*)\)",
                    col_prov.provenance_sql,
                    re.IGNORECASE | re.DOTALL,
                )
                if over_match:
                    over_txt = over_match.group(1).lower()
                    for cand in cols_info:
                        if cand.name and re.search(
                            r"\b" + re.escape(cand.name) + r"\b", over_txt
                        ):
                            protected_cols.add(cand.name)

        upgraded: List[_ColumnProvenance] = []
        for col_prov in cols_info:
            if (
                not col_prov.is_tie_sensitive
                and col_prov.name
                and col_prov.name not in protected_cols
            ):
                upgraded.append(
                    _ColumnProvenance(
                        name=col_prov.name,
                        is_tie_sensitive=True,
                        raw_expr=col_prov.raw_expr,
                        provenance_sql=col_prov.provenance_sql,
                        is_outer_unfiltered_rownum=False,
                        allows_null_shift=True,
                    )
                )
            else:
                upgraded.append(col_prov)
        alias_to_cols[target] = upgraded

    return alias_to_cols


def _analyze_query_columns(
    sql: str,
    outer_ctes: Optional[Dict[str, List[_ColumnProvenance]]] = None,
    is_outer: bool = True,
) -> List[_ColumnProvenance]:
    """Analyze a SQL query and return provenance metadata for each output column.

    Args:
        sql: SQL query string to analyze.
        outer_ctes: Optional enclosing CTE column map for nested subqueries.
        is_outer: Whether `sql` is the outermost query returning final rows.

    Returns:
        List of `_ColumnProvenance` entries corresponding to the query's SELECT list.
    """
    cleaned = _strip_sql_comments(sql)
    known_ctes: Dict[str, List[_ColumnProvenance]] = dict(outer_ctes or {})
    ctes, main_sql = _split_with_and_main_query(cleaned)
    for cte_name, cte_body in ctes:
        known_ctes[cte_name] = _analyze_query_columns(
            cte_body, known_ctes, is_outer=False
        )

    branches = _split_top_level_keyword(
        main_sql, r"\bUNION(?:\s+ALL|\s+DISTINCT)?\b"
    )
    is_single_outer = is_outer and len(branches) == 1
    branch_results: List[List[_ColumnProvenance]] = []

    for branch in branches:
        select_items, from_clause = _extract_select_items_and_from(branch)
        alias_to_cols = _extract_derived_subqueries_and_aliases(
            from_clause, known_ctes
        )

        # Expand `SELECT *` or `SELECT alias.*` when backed by an analyzed CTE/subquery.
        expanded_items: List[Tuple[str, Optional[_ColumnProvenance]]] = []
        for item in select_items:
            star_match = re.match(
                r"^(?:(`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*)?\*\s*$",
                item.strip(),
            )
            if star_match:
                qual = (
                    _unquote_ident(star_match.group(1))
                    if star_match.group(1)
                    else ""
                )
                if qual and qual in alias_to_cols:
                    for col_prov in alias_to_cols[qual]:
                        expanded_items.append((col_prov.raw_expr, col_prov))
                    continue
                from_first = re.search(
                    r"\bFROM\s+(`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)",
                    from_clause,
                    flags=re.IGNORECASE,
                )
                if (
                    not qual
                    and from_first
                    and _unquote_ident(from_first.group(1)) in alias_to_cols
                ):
                    for col_prov in alias_to_cols[
                        _unquote_ident(from_first.group(1))
                    ]:
                        expanded_items.append((col_prov.raw_expr, col_prov))
                    continue
            expanded_items.append((item, None))

        # Index upstream tie-sensitive columns by `(alias, col_name)` and bare `col_name`.
        ts_qualified: Dict[Tuple[str, str], _ColumnProvenance] = {}
        ts_unqualified: Dict[str, _ColumnProvenance] = {}
        for als, clist in alias_to_cols.items():
            for col_prov in clist:
                if col_prov.is_tie_sensitive and col_prov.name:
                    ts_qualified[(als, col_prov.name)] = col_prov
                    ts_unqualified[col_prov.name] = col_prov

        cols_out: List[_ColumnProvenance] = []
        for item, pre_resolved in expanded_items:
            if pre_resolved is not None:
                cols_out.append(
                    _ColumnProvenance(
                        name=pre_resolved.name,
                        is_tie_sensitive=pre_resolved.is_tie_sensitive,
                        raw_expr=pre_resolved.raw_expr,
                        provenance_sql=pre_resolved.provenance_sql,
                        is_outer_unfiltered_rownum=False,
                        allows_null_shift=pre_resolved.allows_null_shift,
                    )
                )
                continue

            cname = _extract_output_col_name(item)
            body = _strip_outer_alias(item)
            is_ts, null_shift = _expr_has_tie_sensitive_window(body)
            prov_parts = [body]

            # Trace qualified `alias.col` references to upstream CTE/subquery columns.
            for qm in re.finditer(
                r"(`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
                r"(`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)",
                body,
            ):
                key = (_unquote_ident(qm.group(1)), _unquote_ident(qm.group(2)))
                if key in ts_qualified:
                    is_ts = True
                    null_shift = (
                        null_shift or ts_qualified[key].allows_null_shift
                    )
                    prov_parts.append(ts_qualified[key].provenance_sql)

            # Trace unqualified `col` references when not preceded by `.`.
            if not is_ts and ts_unqualified:
                for um in re.finditer(
                    r"(?<!\.)(`[^`]+`|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)",
                    body,
                ):
                    u_col = _unquote_ident(um.group(1))
                    if u_col in ts_unqualified:
                        is_ts = True
                        null_shift = (
                            null_shift
                            or ts_unqualified[u_col].allows_null_shift
                        )
                        prov_parts.append(ts_unqualified[u_col].provenance_sql)

            cols_out.append(
                _ColumnProvenance(
                    name=cname,
                    is_tie_sensitive=is_ts,
                    raw_expr=item,
                    provenance_sql=" <- ".join(prov_parts),
                    is_outer_unfiltered_rownum=(
                        is_single_outer
                        and _expr_is_direct_unfiltered_rownum(item)
                    ),
                    allows_null_shift=null_shift,
                )
            )
        branch_results.append(cols_out)

    if not branch_results:
        return []

    merged = list(branch_results[0])
    for br in branch_results[1:]:
        if len(br) == len(merged):
            merged = [
                _ColumnProvenance(
                    name=merged[i].name or br[i].name,
                    is_tie_sensitive=(
                        merged[i].is_tie_sensitive or br[i].is_tie_sensitive
                    ),
                    raw_expr=f"{merged[i].raw_expr} UNION {br[i].raw_expr}",
                    provenance_sql=(
                        f"{merged[i].provenance_sql} UNION {br[i].provenance_sql}"
                    ),
                    is_outer_unfiltered_rownum=False,
                    allows_null_shift=(
                        merged[i].allows_null_shift
                        or br[i].allows_null_shift
                        or True
                    ),
                )
                for i in range(len(merged))
            ]
    return merged


def _has_uncast_pg_division(src_prov_sql: str, dest_prov_sql: str) -> bool:
    """Return True if MySQL uses `/` division and PostgreSQL uses `/` without a float/decimal cast.

    In MySQL, `/` performs floating/decimal division (`5 / 2 = 2.5`), whereas in
    PostgreSQL `/` on integer operands truncates (`5 / 2 = 2`) unless cast to a
    floating/numeric type or multiplied/divided by a decimal literal.

    Args:
        src_prov_sql: Provenance SQL chain for the column in MySQL.
        dest_prov_sql: Provenance SQL chain for the column in PostgreSQL.

    Returns:
        True if the PostgreSQL expression has `/` division with no cast or float literal.
    """
    src_clean = re.sub(r"'[^']*'", "''", src_prov_sql)
    dest_clean = re.sub(r"'[^']*'", "''", dest_prov_sql)
    if "/" not in src_clean or "/" not in dest_clean:
        return False
    has_pg_cast = bool(
        re.search(
            r"::\s*(?:DOUBLE\s+PRECISION|NUMERIC|DECIMAL|FLOAT[48]?|REAL)\b"
            r"|\bCAST\s*\([^()]*\bAS\s+(?:DOUBLE\s+PRECISION|NUMERIC|DECIMAL|FLOAT[48]?|REAL)\b"
            r"|\b[0-9]+\.[0-9]+\b",
            dest_clean,
            re.IGNORECASE,
        )
    )
    return not has_pg_cast


def _identify_tie_sensitive_columns(
    source_query: Optional[str],
    converted_query: Optional[str],
    ncols: int,
) -> Dict[int, Tuple[Optional[_ColumnProvenance], Optional[_ColumnProvenance]]]:
    """Identify 0-based output column indices proven by SQL to be tie-sensitive.

    When both `source_query` and `converted_query` parse to `ncols` output
    columns, a column index is accepted as tie-sensitive only if BOTH queries
    classify that column as tie-sensitive.

    Args:
        source_query: Original MySQL query string.
        converted_query: Candidate PostgreSQL query string.
        ncols: Number of columns in the executed `QueryResult`.

    Returns:
        Dict mapping eligible tie-sensitive column index `i` to
        `(src_col_provenance, dest_col_provenance)`.
    """
    if not source_query and not converted_query:
        return {}

    src_cols = (
        _analyze_query_columns(source_query, is_outer=True)
        if source_query
        else None
    )
    dest_cols = (
        _analyze_query_columns(converted_query, is_outer=True)
        if converted_query
        else None
    )

    src_ok = src_cols is not None and len(src_cols) == ncols
    dest_ok = dest_cols is not None and len(dest_cols) == ncols
    if not src_ok and not dest_ok:
        return {}

    eligible: Dict[
        int, Tuple[Optional[_ColumnProvenance], Optional[_ColumnProvenance]]
    ] = {}
    for idx in range(ncols):
        if src_ok and dest_ok:
            is_ts = (
                src_cols[idx].is_tie_sensitive
                and dest_cols[idx].is_tie_sensitive
            )
        elif dest_ok:
            is_ts = dest_cols[idx].is_tie_sensitive
        else:
            is_ts = src_cols[idx].is_tie_sensitive

        if is_ts:
            eligible[idx] = (
                src_cols[idx] if src_ok else None,
                dest_cols[idx] if dest_ok else None,
            )
    return eligible


class MySQLToPGQueryConverter(QueryConverter):
    """Convert SQL queries from MySQL to PostgreSQL dialect."""

    def __init__(self, allow_tie_equivalence: bool = True):
        """Initialize the MySQL to PostgreSQL query converter.

        Args:
            allow_tie_equivalence: Whether `compare_results` should accept
                tie-equivalent row ordering and verified tie-sensitive window
                results after running the strict row-by-row check. Defaults to True.
        """
        super().__init__("mysql", "postgresql")
        self.allow_tie_equivalence = allow_tie_equivalence

    def get_conversion_prompt(
        self,
        source_schema: str,
        dest_schema: str,
        source_query: str,
        converted_query: Optional[str] = None,
        source_result: Optional[QueryResult] = None,
        dest_result: Optional[QueryResult] = None,
        attempt: int = 1,
        max_attempts: int = 10,
        attempt_history: Optional[List[Dict]] = None,
    ) -> str:
        """Build the ReAct-style prompt for MySQL to PostgreSQL query conversion.

        Args:
            source_schema: MySQL DDL schema text.
            dest_schema: PostgreSQL DDL schema text.
            source_query: Original MySQL query string.
            converted_query: Most recent candidate PostgreSQL query string.
            source_result: Execution result of `source_query` on MySQL.
            dest_result: Execution result of `converted_query` on PostgreSQL.
            attempt: Current 1-based attempt number.
            max_attempts: Maximum number of conversion attempts allowed.
            attempt_history: List of prior attempt dicts with query, result, and notes.

        Returns:
            Prompt string for the LLM conversion agent.
        """
        prompt = f"""You are an expert SQL query converter (ReAct agent) specializing in MySQL to PostgreSQL migrations.

**YOUR GOAL:** Convert a MySQL query to PostgreSQL and ensure results are SEMANTICALLY EQUIVALENT.

**MATCHING CRITERIA (for result_matched):**
- Same number of rows
- Same data values (meaning-wise equivalent)
- Allow precision differences (e.g., 0.5 vs 0.500000)
- Allow type differences (e.g., INTEGER vs DOUBLE PRECISION) if values match

**CONVERSION RULES FOR POSTGRESQL:**

1. **Column and Table Names**: MUST use double quotes
   - MySQL: `Table`.`Column Name` → PostgreSQL: "Table"."Column Name"

2. **Type Casting**: Use PostgreSQL casting
   - MySQL: CAST(x AS REAL) → PostgreSQL: x::DOUBLE PRECISION

3. **NULL Handling**: PostgreSQL requires explicit NULLS FIRST/LAST in ORDER BY
   - MySQL: ORDER BY column ASC → PostgreSQL: ORDER BY column ASC NULLS LAST

4. **Division**: PostgreSQL integer division returns integer
   - MySQL: col1 / col2 (auto-converts to REAL)
   - PostgreSQL: col1::DOUBLE PRECISION / col2 (for real division)

5. **Boolean Values**:
   - MySQL: 0/1 → PostgreSQL: Use explicit comparison (= 1, = 0)

===== SCHEMAS =====

SOURCE (MySQL):
{source_schema}

DESTINATION (PostgreSQL):
{dest_schema}

===== SOURCE QUERY (MySQL) =====
{source_query}
"""

        # Include the source execution summary when available.
        if source_result:
            prompt += f"""
===== SOURCE EXECUTION RESULT =====
{source_result.get_summary()}
"""

        # Include the history of prior attempts so the agent does not repeat failed queries.
        if attempt_history:
            prompt += "\n===== PREVIOUS ATTEMPTS =====\n"
            for hist in attempt_history:
                prompt += f"\n--- Attempt {hist['attempt']} ---\n"
                prompt += f"Query:\n{hist['query']}\n\n"
                result = hist["result"]
                if result.error:
                    prompt += f"Result: ERROR - {result.error}\n"
                else:
                    prompt += f"Result: {result.total_rows} rows\n"
                prompt += f"Notes: {hist['notes']}\n"
            prompt += "\n"

        # Include the current candidate query and comparison result if present.
        if converted_query:
            prompt += f"""
===== CURRENT CONVERSION ATTEMPT =====
{converted_query}
"""

            if dest_result:
                prompt += f"""
===== DESTINATION EXECUTION RESULT =====
{dest_result.get_summary() if not dest_result.error else f"ERROR: {dest_result.error}"}
"""

                if source_result and not dest_result.error:
                    prompt += f"""
===== COMPARISON ANALYSIS =====
- Source rows: {source_result.total_rows}
- Destination rows: {dest_result.total_rows}
"""
                    if source_result.total_rows == dest_result.total_rows:
                        prompt += "✓ Row counts MATCH\n"
                        matches, reason = self.compare_results(
                            source_result,
                            dest_result,
                            source_query=source_query,
                            converted_query=converted_query,
                        )
                        if matches:
                            prompt += f"✓ Data MATCHES: {reason}\n"
                        else:
                            prompt += f"✗ Data MISMATCH: {reason}\n"
                    else:
                        prompt += "✗ Row counts MISMATCH\n"

        prompt += f"""
===== YOUR TASK (Attempt {attempt}/{max_attempts}) =====

**Read the information above and decide what to do:**

1. If results MATCH (same rows, meaning-wise same data) → Declare SUCCESS with "result_matched"
2. If you need to try a different conversion → Provide a new "converted_query" to test
3. If conversion is impossible (schema issues, data loss) → Declare "unable_to_match"

**OUTPUT FORMAT (JSON only, no markdown):**

Option 1 - Results match:
{{
  "conversion_finished": "result_matched",
  "notes": "Explain why results match (e.g., same rows, data matches meaning-wise)"
}}

Option 2 - Try new conversion:
{{
  "converted_query": "SELECT ...",
  "notes": "Explain what you're fixing/trying (e.g., fixing NULL ordering, adding type cast)"
}}

Option 3 - Cannot match:
{{
  "conversion_finished": "unable_to_match",
  "notes": "Explain why impossible (e.g., schema mismatch, data not available)"
}}

**CRITICAL REMINDERS:**
1. Your #1 priority is "result_matched". Try VERY HARD to fix any issues!
2. Review ALL previous attempts above to avoid repeating the same failed conversions.
3. If you keep getting the same error, try a DIFFERENT approach (e.g., use CTE, subquery, different casting).
4. Precision/type differences are OK if data meaning is the same.

Output your decision now:
"""

        return prompt

    def compare_results(
        self,
        source_result: QueryResult,
        dest_result: QueryResult,
        source_query: Optional[str] = None,
        converted_query: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Compare MySQL and PostgreSQL query results using mechanical checks.

        Checks in order:
          1. Strict row-by-row check in the returned order (allowing `1e-4`
             decimal rounding tolerance).
          2. Reordered rows on ties: If row and column counts match, checks if
             sorting both tables produces the exact same rows across all columns
             (zero columns dropped).
          3. Verified tie-sensitive window/rank columns: Only runs when
             `source_query` or `converted_query` is provided. A differing column
             may be set aside ONLY if SQL provenance proves that specific output
             column derives from a physical-order window (`ROWS` frame,
             `ROW_NUMBER`, `NTILE`, `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`,
             `NTH_VALUE`) or top-1 rank payload AND value/division sanity guards
             pass (preventing integer-division `AVG` or collation bugs from passing).

        Args:
            source_result: Result from executing the MySQL query.
            dest_result: Result from executing the PostgreSQL query.
            source_query: Optional original MySQL query string used to verify
                per-column window/rank SQL provenance in Check 3.
            converted_query: Optional candidate PostgreSQL query string used to
                verify per-column window/rank SQL provenance in Check 3.

        Returns:
            Tuple of `(matches, reason)` indicating whether the results match.
        """
        if source_result.error and dest_result.error:
            return (
                False,
                f"Both queries failed. Source: {source_result.error}, Dest: {dest_result.error}",
            )
        if source_result.error:
            return False, f"Source query error: {source_result.error}"
        if dest_result.error:
            return False, f"Destination query error: {dest_result.error}"

        if source_result.total_rows != dest_result.total_rows:
            return (
                False,
                f"Row count mismatch: source={source_result.total_rows}, dest={dest_result.total_rows}",
            )

        if len(source_result.columns) != len(dest_result.columns):
            return (
                False,
                f"Column count mismatch: source={len(source_result.columns)}, dest={len(dest_result.columns)}",
            )

        if len(source_result.rows) != len(dest_result.rows):
            return (
                False,
                f"Data row mismatch: source={len(source_result.rows)}, dest={len(dest_result.rows)}",
            )

        if source_result.total_rows == 0:
            return True, "All checks passed: both queries returned 0 rows"

        # 1. Original row-by-row check in returned order.
        first_mismatch_reason = None
        for i, (src_row, dest_row) in enumerate(
            zip(source_result.rows, dest_result.rows)
        ):
            if len(src_row) != len(dest_row):
                return False, f"Row {i + 1} column count mismatch"
            for j, (src_val, dest_val) in enumerate(zip(src_row, dest_row)):
                if not self._values_equal(src_val, dest_val):
                    first_mismatch_reason = (
                        f"Row {i + 1}, column {j + 1} mismatch: "
                        f"source={src_val}, dest={dest_val}"
                    )
                    break
            if first_mismatch_reason is not None:
                break

        if first_mismatch_reason is None:
            return (
                True,
                "All checks passed: row counts match, column counts match, data matches",
            )

        if not self.allow_tie_equivalence:
            return False, first_mismatch_reason

        ncols = len(source_result.columns)
        all_cols = list(range(ncols))

        # 2. Check if sorting both tables produces the exact same rows across ALL columns.
        if self._rows_match_on_columns(
            source_result.rows, dest_result.rows, all_cols
        ):
            return (
                True,
                "All checks passed (same rows reordered): exact same set of rows "
                "across all columns (only row order differs on ties)",
            )

        # 3. Tightened window/rank tie check: ONLY columns proven by SQL provenance
        # to derive from a physical-order window/top-1 construct may be set aside,
        # and every set-aside column must pass value-domain and division guards.
        if ncols >= 2 and (source_query or converted_query):
            eligible_ts = _identify_tie_sensitive_columns(
                source_query, converted_query, ncols
            )
            if not eligible_ts:
                return False, first_mismatch_reason

            diff_cols = [
                col_idx
                for col_idx in all_cols
                if not self._column_values_match(
                    source_result.rows, dest_result.rows, col_idx
                )
            ]

            # If any column whose multiset of values differs is NOT tie-sensitive
            # (e.g., an AVG/COUNT/SUM or base column), reject immediately.
            if not set(diff_cols).issubset(eligible_ts.keys()):
                return False, first_mismatch_reason

            # Filter eligible tie-sensitive columns through per-column sanity checks.
            valid_ts_cols = [
                col_idx
                for col_idx in sorted(eligible_ts.keys())
                if self._dropped_column_passes_sanity(
                    source_result.rows,
                    dest_result.rows,
                    col_idx,
                    col_idx not in diff_cols,
                    eligible_ts[col_idx][0],
                    eligible_ts[col_idx][1],
                )
            ]
            if not set(diff_cols).issubset(valid_ts_cols):
                return False, first_mismatch_reason

            min_base_cols = 1 if ncols == 2 else max(2, (ncols + 1) // 2)
            max_drop = min(3, len(valid_ts_cols), ncols - min_base_cols)
            for num_drop in range(max(1, len(diff_cols)), max_drop + 1):
                for dropped in self._column_combinations(
                    valid_ts_cols, num_drop
                ):
                    if not set(diff_cols).issubset(dropped):
                        continue
                    rem_cols = [c for c in all_cols if c not in dropped]
                    if len(rem_cols) < min_base_cols:
                        continue
                    if self._rows_match_on_columns(
                        source_result.rows, dest_result.rows, rem_cols
                    ):
                        dropped_names = [
                            dest_result.columns[c] for c in dropped
                        ]
                        return (
                            True,
                            f"All checks passed (window/rank tie-equivalent): all "
                            f"{len(rem_cols)}/{ncols} base columns match 100%; "
                            f"only verified tie-dependent column(s) {dropped_names} differ",
                        )

        return False, first_mismatch_reason

    def _dropped_column_passes_sanity(
        self,
        src_rows: Sequence[Tuple],
        dest_rows: Sequence[Tuple],
        col_idx: int,
        multiset_matches: bool,
        src_prov: Optional[_ColumnProvenance],
        dest_prov: Optional[_ColumnProvenance],
    ) -> bool:
        """Verify value-level and SQL division invariants on a tie-sensitive column.

        Guards enforced:
          1. Direct unfiltered `ROW_NUMBER()` / `NTILE()` columns in the outer
             `SELECT` must preserve their exact multiset of rank integers.
          2. NULL counts must match unless the expression uses `LAG`/`LEAD` or
             top-1 `LEFT JOIN` / scalar `LIMIT 1` where boundary NULLs can shift.
          3. Integer-truncation guard: rejects if MySQL produced fractional
             decimals while PostgreSQL produced only exact integers.
          4. Uncast division guard: rejects if the column values differ and
             PostgreSQL performs `/` division without any float/numeric cast.

        Args:
            src_rows: Rows from the MySQL source query.
            dest_rows: Rows from the PostgreSQL destination query.
            col_idx: Zero-based index of the candidate tie-sensitive column.
            multiset_matches: True if `_column_values_match` passed on `col_idx`.
            src_prov: Column provenance from `source_query` if available.
            dest_prov: Column provenance from `converted_query` if available.

        Returns:
            True if all sanity invariants hold for `col_idx`.
        """
        is_unfiltered_rn = bool(
            (dest_prov and dest_prov.is_outer_unfiltered_rownum)
            or (src_prov and src_prov.is_outer_unfiltered_rownum)
        )
        if is_unfiltered_rn and not multiset_matches:
            return False

        allows_null_shift = bool(
            (dest_prov and dest_prov.allows_null_shift)
            or (src_prov and src_prov.allows_null_shift)
        )
        src_nulls = sum(1 for r in src_rows if r[col_idx] is None)
        dest_nulls = sum(1 for r in dest_rows if r[col_idx] is None)
        if not allows_null_shift and src_nulls != dest_nulls:
            return False

        if not multiset_matches:
            if (
                src_prov is not None
                and dest_prov is not None
                and _has_uncast_pg_division(
                    src_prov.provenance_sql, dest_prov.provenance_sql
                )
            ):
                return False

            # Check whether MySQL has non-integer decimals while PostgreSQL truncated to integers.
            src_nums: List[float] = []
            dest_nums: List[float] = []
            src_non_null = len(src_rows) - src_nulls
            dest_non_null = len(dest_rows) - dest_nulls
            try:
                src_nums = [
                    float(str(r[col_idx]).strip())
                    for r in src_rows
                    if r[col_idx] is not None
                ]
                dest_nums = [
                    float(str(r[col_idx]).strip())
                    for r in dest_rows
                    if r[col_idx] is not None
                ]
            except (ValueError, TypeError):
                src_nums = []
                dest_nums = []

            if (
                src_non_null > 0
                and dest_non_null > 0
                and len(src_nums) == src_non_null
                and len(dest_nums) == dest_non_null
            ):
                src_has_fraction = any(
                    not math.isnan(v) and abs(v - round(v)) > 1e-3
                    for v in src_nums
                )
                dest_all_int = all(
                    not math.isnan(v) and abs(v - round(v)) <= 1e-9
                    for v in dest_nums
                )
                if src_has_fraction and dest_all_int:
                    return False

        return True

    def _values_equal(
        self,
        val1: Any,
        val2: Any,
        rel_tol: float = 1e-4,
        abs_tol: float = 1e-4,
    ) -> bool:
        """Return True if two cell values are equal within numeric/case tolerance.

        Args:
            val1: Cell value from MySQL result.
            val2: Cell value from PostgreSQL result.
            rel_tol: Relative tolerance for floating-point comparison.
            abs_tol: Absolute tolerance for floating-point comparison.

        Returns:
            True if the two values match.
        """
        if val1 is None and val2 is None:
            return True
        if val1 is None or val2 is None:
            return False

        str1 = str(val1).strip()
        str2 = str(val2).strip()

        if str1.lower() == str2.lower():
            return True

        try:
            num1 = float(str1)
            num2 = float(str2)
            if math.isnan(num1) and math.isnan(num2):
                return True
            return math.isclose(num1, num2, rel_tol=rel_tol, abs_tol=abs_tol)
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _canonicalize_value(
        val: Any, round_digits: Optional[int] = 3
    ) -> Tuple[str, Any]:
        """Convert a cell value into a sortable key tuple for row comparison.

        Args:
            val: Raw cell value from a database driver.
            round_digits: Optional decimal places to round numeric sort keys to
                so values equal within `1e-4` group together; pass `None` for
                exact float ordering on 1-D columns.

        Returns:
            Tuple of `(type_tag, normalized_value)` that can be sorted.
        """
        if val is None:
            return ("N", "")
        text = str(val).strip()
        try:
            num = float(text)
            if math.isnan(num):
                return ("N", "nan")
            if round_digits is None:
                return ("F", num)
            return ("F", round(num, round_digits))
        except (ValueError, TypeError):
            return ("S", text.lower())

    def _column_values_match(
        self,
        src_rows: Sequence[Tuple],
        dest_rows: Sequence[Tuple],
        col_idx: int,
    ) -> bool:
        """Return True if a single column has the same values in both tables when sorted.

        Args:
            src_rows: Rows from the source database.
            dest_rows: Rows from the destination database.
            col_idx: Zero-based column index to compare.

        Returns:
            True if the sorted column values match element-by-element.
        """
        src_sorted = [
            r[col_idx]
            for _, r in sorted(
                zip(
                    [
                        self._canonicalize_value(r[col_idx], round_digits=None)
                        for r in src_rows
                    ],
                    src_rows,
                ),
                key=lambda item: item[0],
            )
        ]
        dest_sorted = [
            r[col_idx]
            for _, r in sorted(
                zip(
                    [
                        self._canonicalize_value(r[col_idx], round_digits=None)
                        for r in dest_rows
                    ],
                    dest_rows,
                ),
                key=lambda item: item[0],
            )
        ]
        return all(
            self._values_equal(a, b) for a, b in zip(src_sorted, dest_sorted)
        )

    def _rows_match_on_columns(
        self,
        src_rows: Sequence[Tuple],
        dest_rows: Sequence[Tuple],
        cols: Sequence[int],
    ) -> bool:
        """Return True if sorting both tables on `cols` yields the same rows.

        Tries multiple coarse float sort-bucketing precisions (`4`, `3`, and `2`
        decimals) so two floats within `1e-4` that straddle one rounding boundary
        do not swap adjacent tied rows, while still verifying every cell pair with
        `_values_equal` (`1e-4`).

        Args:
            src_rows: Rows from the source database.
            dest_rows: Rows from the destination database.
            cols: Sequence of column indices to compare.

        Returns:
            True if every row in `src_rows` matches `dest_rows` on `cols` after sorting.
        """
        if not cols:
            return False

        for digits in (4, 3, 2):
            src_sorted = [
                tuple(r[c] for c in cols)
                for _, r in sorted(
                    zip(
                        [
                            tuple(
                                self._canonicalize_value(
                                    r[c], round_digits=digits
                                )
                                for c in cols
                            )
                            for r in src_rows
                        ],
                        src_rows,
                    ),
                    key=lambda item: item[0],
                )
            ]
            dest_sorted = [
                tuple(r[c] for c in cols)
                for _, r in sorted(
                    zip(
                        [
                            tuple(
                                self._canonicalize_value(
                                    r[c], round_digits=digits
                                )
                                for c in cols
                            )
                            for r in dest_rows
                        ],
                        dest_rows,
                    ),
                    key=lambda item: item[0],
                )
            ]
            if all(
                all(self._values_equal(a, b) for a, b in zip(src_r, dest_r))
                for src_r, dest_r in zip(src_sorted, dest_sorted)
            ):
                return True
        return False

    @staticmethod
    def _column_combinations(
        cols: Sequence[int], k: int
    ) -> List[Tuple[int, ...]]:
        """Return all k-element index combinations from `cols` in ascending order.

        Args:
            cols: Available column indices.
            k: Number of columns per combination (1, 2, or 3).

        Returns:
            List of index tuples of length `k`.
        """
        if k == 1:
            return [(c,) for c in cols]
        if k == 2:
            return [
                (cols[i], cols[j])
                for i in range(len(cols))
                for j in range(i + 1, len(cols))
            ]
        if k == 3:
            return [
                (cols[i], cols[j], cols[m])
                for i in range(len(cols))
                for j in range(i + 1, len(cols))
                for m in range(j + 1, len(cols))
            ]
        return []
