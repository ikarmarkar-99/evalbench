"""MySQL to PostgreSQL SQL query converter with tie-aware result comparison.

This module contains:
  - `MySQLToPGQueryConverter`: Builds the prompt for converting MySQL queries to
    PostgreSQL and implements `compare_results` to compare two result tables
    across 100% of columns:
      1. Check 1: Exact row-by-row match in the returned order (whole numbers
         and text exactly; other numbers to one part in a billion, one part in
         a million for MySQL floating-point values, or equal once the
         PostgreSQL value is rounded the way MySQL rounded its value).
      2. Check 2: Same exact rows across all columns, where rows may only swap
         places with rows they are tied with on the source query's top-level
         ORDER BY keys (so the sort order itself is always checked, and zero
         columns are ever ignored or dropped).
  - Helpers that read the source query's top-level ORDER BY keys and SELECT
    list (`_mask_sql`, `_split_top_level`, `_top_level_order_keys`,
    `_select_items`) so Check 2 knows which output columns are sort keys.
"""

import decimal
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .base import QueryConverter, QueryResult


# A number written the way a database prints it: no leading zeros (so text
# codes like '007' or '02139' are never treated as numbers).
_CANONICAL_NUMBER_RE = re.compile(
    r"^-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?$"
)
# Relative tolerance for non-whole numbers: they must agree to one part in a
# billion (with the same 1e-9 as an absolute floor near zero), matching the
# existing converters' 1e-9 while staying fair to very large values.
_REL_TOL = 1e-9
_ABS_TOL = 1e-9
# Values MySQL returns as floating point come from its FLOAT columns (or math
# on them). MySQL FLOAT keeps only about 7 significant digits, while the same
# column may be stored at full precision in PostgreSQL, so these values may
# differ by one part in a million. Exact decimals, counts and IDs never get
# this leeway.
_MYSQL_FLOAT_REL_TOL = 1e-6
# A canonical whole number (no decimal point and no exponent).
_CANONICAL_INT_RE = re.compile(r"^-?(0|[1-9]\d*)$")
# Words that end an ORDER BY clause at the top level of a query.
_ORDER_END_RE = re.compile(
    r"\b(limit|offset|fetch|for\s+update|for\s+share|lock\s+in)\b|;",
    re.IGNORECASE,
)
# Sort modifiers that can follow an ORDER BY key.
_ORDER_SUFFIX_RE = re.compile(
    r"(\s+collate\s+\S+)?(\s+(asc|desc))?(\s+nulls\s+(first|last))?\s*$",
    re.IGNORECASE,
)
# A plain (optionally table-qualified) column reference such as `t`.`col`.
_SIMPLE_REF_RE = re.compile(r"^[\w$]+(\.[\w$]+)*$")


def _mask_sql(sql: str, mask_parens: bool) -> str:
    """Returns `sql` with quoted text and comments replaced by spaces.

    The result has the same length as `sql`, so positions found in it can be
    used to slice `sql`. Handles '...' (with '' and backslash escapes), "...",
    `...`, `--` and `#` line comments, and /* */ comments.

    Args:
        sql: A SQL query.
        mask_parens: If True, text inside parentheses is also replaced by
            spaces (the parentheses themselves too), leaving only the
            top level of the query visible.

    Returns:
        The masked query.
    """
    out = []
    i, n, depth = 0, len(sql), 0
    while i < n:
        ch = sql[i]
        # Line comments: blank out to the end of the line.
        if sql.startswith("--", i) or ch == "#":
            j = sql.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
            continue
        # Block comments: blank out to the closing */.
        if sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(" " * (j - i))
            i = j
            continue
        # Quoted text or identifiers: keep the quotes, blank the inside.
        if ch in "'\"`":
            j = i + 1
            while j < n:
                if sql[j] == "\\" and ch == "'":
                    j += 2
                    continue
                if sql[j] == ch:
                    if j + 1 < n and sql[j + 1] == ch:
                        j += 2
                        continue
                    break
                j += 1
            j = min(j, n - 1)
            inner = " " * (j - i - 1)
            if mask_parens and depth > 0:
                out.append(" " * (j - i + 1))
            else:
                out.append(ch + inner + sql[j] if j > i else ch)
            i = j + 1
            continue
        if mask_parens and ch == "(":
            depth += 1
            out.append(" ")
        elif mask_parens and ch == ")":
            depth = max(depth - 1, 0)
            out.append(" ")
        else:
            out.append(" " if (mask_parens and depth > 0) else ch)
        i += 1
    return "".join(out)


def _split_top_level(text: str) -> List[str]:
    """Splits `text` on commas that are not inside parentheses or quotes."""
    masked = _mask_sql(text, mask_parens=True)
    parts, start = [], 0
    for i, ch in enumerate(masked):
        if ch == ",":
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _top_level_order_keys(sql: str) -> Optional[List[str]]:
    """Returns the sort keys of the query's outermost ORDER BY, or None.

    ORDER BY clauses inside parentheses (subqueries, CTEs, window functions)
    are ignored; only the one that sorts the final result counts. Each key is
    returned without its ASC/DESC, NULLS FIRST/LAST or COLLATE suffix.

    Args:
        sql: A SQL query.

    Returns:
        The list of key expressions, or None if the final result is not
        sorted.
    """
    masked = _mask_sql(sql, mask_parens=True)
    matches = list(re.finditer(r"\border\s+by\b", masked, re.IGNORECASE))
    if not matches:
        return None
    start = matches[-1].end()
    end_match = _ORDER_END_RE.search(masked, start)
    end = end_match.start() if end_match else len(sql)
    keys = []
    for key in _split_top_level(sql[start:end]):
        key = _ORDER_SUFFIX_RE.sub("", key).strip()
        if key:
            keys.append(key)
    return keys or None


def _normalize_expr(text: str) -> str:
    """Lower-cases `text`, drops quote characters and extra spaces."""
    text = text.replace("`", "").replace('"', "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s*([(),.])\s*", r"\1", text)


def _select_items(sql: str) -> List[Tuple[str, Optional[str]]]:
    """Returns `(expression, alias)` for each item of the outermost SELECT.

    For a UNION, this is the first branch, which names the output columns.

    Args:
        sql: A SQL query.

    Returns:
        A list of `(normalized expression, normalized alias or None)`.
    """
    masked = _mask_sql(sql, mask_parens=True)
    sel = re.search(r"\bselect\b(\s+(distinct|all)\b)?", masked, re.IGNORECASE)
    if not sel:
        return []
    frm = re.search(
        r"\b(from|union|intersect|except|order\s+by|limit)\b",
        masked[sel.end():],
        re.IGNORECASE,
    )
    end = sel.end() + frm.start() if frm else len(sql)
    items = []
    for item in _split_top_level(sql[sel.end():end]):
        m = re.match(
            r"^(?P<expr>.*?)\s+(as\s+)?(?P<alias>[`\"]?[\w$ ]+[`\"]?)$",
            item,
            re.IGNORECASE | re.DOTALL,
        )
        expr, alias = item, None
        if m and not re.search(r"[-+*/|=<>,(]\s*$", m.group("expr")):
            alias_text = m.group("alias").strip()
            # A bare trailing word is only an alias if it is quoted or the
            # expression before it is complete (ends in a name, quote or ")").
            if re.search(r"\bas\s+\S+$", item, re.IGNORECASE) or re.search(
                r"[\w`\")]$", m.group("expr").strip()
            ):
                expr, alias = m.group("expr"), _normalize_expr(alias_text)
        items.append((_normalize_expr(expr), alias))
    return items


class MySQLToPGQueryConverter(QueryConverter):
    """Convert SQL queries from MySQL to PostgreSQL."""

    def __init__(self, allow_tie_equivalence: bool = True):
        """Initialize the MySQL to PostgreSQL query converter.

        Args:
            allow_tie_equivalence: Whether to treat two queries as matching when
                they return the exact same rows in a different tie order, or
                when they match across 100% of columns after breaking ties the
                same way in both queries. Defaults to True.
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
        """Build the LLM prompt for converting a MySQL query to PostgreSQL.

        Args:
            source_schema: Text of the MySQL table definitions.
            dest_schema: Text of the PostgreSQL table definitions.
            source_query: Original MySQL query string.
            converted_query: Most recent candidate PostgreSQL query string.
            source_result: Result of running `source_query` on MySQL.
            dest_result: Result of running `converted_query` on PostgreSQL.
            attempt: Current 1-based attempt number.
            max_attempts: Maximum number of conversion attempts allowed.
            attempt_history: List of earlier attempts with their SQL, result, and notes.

        Returns:
            Prompt string sent to the LLM conversion agent.
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

3. **NULL Handling**: MySQL puts NULLs FIRST on ASC and LAST on DESC; PostgreSQL does the opposite by default
   - MySQL: ORDER BY column ASC → PostgreSQL: ORDER BY column ASC NULLS FIRST
   - MySQL: ORDER BY column DESC → PostgreSQL: ORDER BY column DESC NULLS LAST

3b. **Text Sort Order**: MySQL's text order differs from PostgreSQL's default, which ignores spaces and
   punctuation (MySQL: 'Adv Comp' < 'Advanced', 'D-Lab' < 'Defects'; PostgreSQL default: the reverse).
   When sorting by a text column (in a final ORDER BY or inside OVER (... ORDER BY ...)), pick the
   PostgreSQL sort expression from the MySQL table's collation shown in the source schema:
   - `COLLATE=utf8mb4_0900_ai_ci` (or any `utf8mb4_0900_*`) or `*_unicode_ci`:
     MySQL: ORDER BY title → PostgreSQL: ORDER BY "title" COLLATE "und-x-icu" ASC NULLS FIRST
   - `*_general_ci`, or a table with only `DEFAULT CHARSET=utf8mb3` / `latin1` and no COLLATE
     (their default collations are `utf8mb3_general_ci` / `latin1_swedish_ci`):
     MySQL: ORDER BY title → PostgreSQL: ORDER BY UPPER("title") COLLATE "C" ASC NULLS FIRST
   - Only for text columns (COLLATE on a number column is an error). Do not use ctid or
     hand-written CASE overrides to imitate MySQL's order of tied rows; tied rows may come in any order.

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
                            source_result, dest_result, source_query=source_query
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
    ) -> Tuple[bool, str]:
        """Compare MySQL and PostgreSQL query results across 100% of columns.

        Checks in order:
          1. Check 1 (exact row order): Compares row 1 to row 1, row 2 to row 2,
             and so on across every column (see `_values_equal`: whole numbers
             and text must match exactly; other numbers must agree to one part
             in a billion, or a million for MySQL floating-point values, or the
             MySQL value must be the PostgreSQL value rounded to the MySQL
             value's number of decimal places).
          2. Check 2 (same rows in a different tie order): If
             `allow_tie_equivalence` is True and `source_query` is given, rows
             may only swap places with rows that are tied on the source query's
             top-level ORDER BY keys (see `_order_respecting_match`). If the
             source query has no top-level ORDER BY, row order is undefined and
             any order is accepted. Zero columns are ever ignored or dropped.

        Args:
            source_result: Result from running the MySQL query.
            dest_result: Result from running the PostgreSQL query.
            source_query: The MySQL query that produced `source_result`. Needed
                for Check 2; without it, rows in a different order never match.

        Returns:
            Tuple of `(matches, reason)` indicating whether the results match.
        """
        # If either query failed with a database error, they cannot match.
        if source_result.error and dest_result.error:
            return (
                False,
                f"Both queries failed. Source: {source_result.error}, Dest: {dest_result.error}",
            )
        if source_result.error:
            return False, f"Source query error: {source_result.error}"
        if dest_result.error:
            return False, f"Destination query error: {dest_result.error}"

        # Both tables must have the exact same number of rows and columns.
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

        # Check 1: Compare every cell row-by-row in the exact returned order.
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

        # If tie-equivalence is turned off (--strict-row-order), stop here.
        if not self.allow_tie_equivalence:
            return False, first_mismatch_reason

        # Check 2 needs the source query to know which columns set the order.
        if source_query is None:
            return False, first_mismatch_reason

        ok, why = self._order_respecting_match(
            source_query, source_result, dest_result
        )
        if ok:
            return (
                True,
                "All checks passed (same rows reordered): exact same set of rows "
                f"across all columns ({why})",
            )
        return False, f"{first_mismatch_reason}; {why}"

    def _order_respecting_match(
        self,
        source_query: str,
        source_result: QueryResult,
        dest_result: QueryResult,
    ) -> Tuple[bool, str]:
        """Check 2: same rows, where rows may only move within ORDER BY ties.

        If the source query has no top-level ORDER BY, the database may return
        rows in any order, so the two results only need the same rows. If it
        has one, each ORDER BY key is matched to an output column, both results
        are cut into consecutive groups of rows that share the same key values
        (using MySQL's notion of "the same": case-insensitive text, trailing
        spaces ignored), and the groups must appear in the same order with the
        same rows inside. So a reversed sort, different NULL placement or a
        different text sort order is rejected, while a different order among
        tied rows is accepted.

        Args:
            source_query: The MySQL query that produced `source_result`.
            source_result: Result from running the MySQL query.
            dest_result: Result from running the PostgreSQL query.

        Returns:
            `(matches, explanation)`.
        """
        all_cols = list(range(len(source_result.columns)))
        keys = _top_level_order_keys(source_query)
        # No ORDER BY: only the set of rows matters.
        if keys is None:
            if self._rows_match_on_columns(
                source_result.rows, dest_result.rows, all_cols
            ):
                return True, "the source query has no ORDER BY, so row order is free"
            return False, "the rows differ, not just their order"

        key_cols = self._order_key_columns(keys, source_query, source_result)
        if key_cols is None:
            return (
                False,
                "row order differs and the ORDER BY keys are not all output "
                "columns, so the order cannot be checked",
            )

        src_groups = self._tie_groups(source_result.rows, key_cols)
        dest_groups = self._tie_groups(dest_result.rows, key_cols)
        if len(src_groups) != len(dest_groups):
            return False, "the rows are not in the same ORDER BY order"
        for (src_key, src_rows), (dest_key, dest_rows) in zip(
            src_groups, dest_groups
        ):
            same_key = all(
                self._keys_tied(a, b, src_vs_dest=True)
                for a, b in zip(src_key, dest_key)
            )
            if (
                not same_key
                or len(src_rows) != len(dest_rows)
                or not self._rows_match_on_columns(src_rows, dest_rows, all_cols)
            ):
                return False, "the rows are not in the same ORDER BY order"
        return True, "only rows tied on the ORDER BY keys changed places"

    @staticmethod
    def _order_key_columns(
        keys: Sequence[str], source_query: str, source_result: QueryResult
    ) -> Optional[List[int]]:
        """Maps each ORDER BY key to the output column it sorts by.

        A key matches a column by position (`ORDER BY 2`), by output column
        name or alias, or by being the same expression as a SELECT item.

        Args:
            keys: ORDER BY keys from `_top_level_order_keys`.
            source_query: The MySQL query.
            source_result: Its result (for the output column names).

        Returns:
            One column index per key, or None if any key is not an output
            column (then the order cannot be checked from the results).
        """
        names = [_normalize_expr(str(c)) for c in source_result.columns]
        items = _select_items(source_query)
        out = []
        for key in keys:
            norm = _normalize_expr(key)
            idx = None
            if re.fullmatch(r"\d+", norm):
                n = int(norm) - 1
                idx = n if 0 <= n < len(names) else None
            # Same expression as a SELECT item (e.g. `t.col` or `SUM(x)`).
            if idx is None:
                for i, (expr, _) in enumerate(items):
                    if expr == norm and i < len(names):
                        idx = i
                        break
            # Output column name or alias (for `t.col`, the name is `col`).
            if idx is None:
                short = norm.split(".")[-1] if _SIMPLE_REF_RE.match(norm) else norm
                for i, name in enumerate(names):
                    if name in (norm, short):
                        idx = i
                        break
            if idx is None:
                for i, (_, alias) in enumerate(items):
                    if alias is not None and alias == norm and i < len(names):
                        idx = i
                        break
            if idx is None:
                return None
            out.append(idx)
        return out

    def _tie_groups(
        self, rows: Sequence[Tuple], key_cols: Sequence[int]
    ) -> List[Tuple[Tuple, List[Tuple]]]:
        """Cuts `rows` into consecutive groups that share the same key values."""
        groups: List[Tuple[Tuple, List[Tuple]]] = []
        for row in rows:
            key = tuple(row[c] for c in key_cols)
            if groups and all(
                self._keys_tied(a, b) for a, b in zip(groups[-1][0], key)
            ):
                groups[-1][1].append(row)
            else:
                groups.append((key, [row]))
        return groups

    @classmethod
    def _keys_tied(cls, val1: Any, val2: Any, src_vs_dest: bool = False) -> bool:
        """True if MySQL would treat two sort-key values as tied.

        MySQL's default text comparison ignores letter case and trailing
        spaces, so 'Apple' and 'apple ' are tied. Numbers within one result are
        tied when they agree to one part in a billion; when comparing a MySQL
        key (`val1`) with a PostgreSQL key (`val2`, `src_vs_dest=True`), the
        rules of `_numbers_match` apply.
        """
        if val1 is None or val2 is None:
            return val1 is None and val2 is None
        num1, num2 = cls._as_number(val1), cls._as_number(val2)
        if num1 is not None and num2 is not None:
            if src_vs_dest:
                return cls._numbers_match(num1, num2, val1, _ABS_TOL)
            return math.isclose(
                float(num1), float(num2), rel_tol=_REL_TOL, abs_tol=_ABS_TOL
            )
        return str(val1).rstrip().lower() == str(val2).rstrip().lower()

    @staticmethod
    def _as_number(val: Any) -> Optional[Any]:
        """Returns `val` as an int or Decimal/float if it is a number, else None.

        Booleans count as 0/1 (MySQL has no boolean type). Text counts as a
        number only if it is written canonically, with no leading zeros, so
        codes like '007' stay text.
        """
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, (int, float, decimal.Decimal)):
            return val
        if isinstance(val, str) and _CANONICAL_NUMBER_RE.match(val):
            if _CANONICAL_INT_RE.match(val):
                return int(val)
            return decimal.Decimal(val)
        return None

    @staticmethod
    def _is_whole(num: Any) -> bool:
        """True for whole-number values: ints and Decimals without decimals."""
        if isinstance(num, int):
            return True
        if isinstance(num, decimal.Decimal) and num.is_finite():
            return num.as_tuple().exponent >= 0
        return False

    @staticmethod
    def _decimal_places(val: Any) -> int:
        """Number of digits after the decimal point MySQL printed for `val`.

        Only exact decimals carry this (MySQL `DECIMAL` results, including
        `AVG` and `/`, come back as `Decimal`, e.g. `Decimal('11.3333')` has 4).
        Floats and whole numbers return 0.
        """
        if isinstance(val, str) and _CANONICAL_NUMBER_RE.match(val):
            val = decimal.Decimal(val)
        if isinstance(val, decimal.Decimal) and val.is_finite():
            return max(-val.as_tuple().exponent, 0)
        return 0

    @classmethod
    def _numbers_match(
        cls, src_num: Any, dest_num: Any, src_raw: Any, epsilon: float
    ) -> bool:
        """Compares a MySQL number with a PostgreSQL number.

        Rules:
          - Both whole numbers: must be exactly equal (compared as integers).
          - Otherwise they match if they agree to one part in a billion
            (`_REL_TOL`, with `epsilon` as the absolute floor near zero), or
            to one part in a million (`_MYSQL_FLOAT_REL_TOL`) when MySQL
            returned the value as floating point (its 7-digit FLOAT columns).
          - Or if the MySQL value is a decimal with N digits after the point
            and it equals the PostgreSQL value rounded (half away from zero)
            to N digits. That covers MySQL showing `AVG` and division to 4
            decimals (11.3333 vs 11.333333...) without allowing any other
            difference: 11.3333 vs 11.3340 still fails, and a whole MySQL
            number (N = 0) is never matched by rounding.

        Args:
            src_num: MySQL value as a number (from `_as_number`).
            dest_num: PostgreSQL value as a number (from `_as_number`).
            src_raw: The original MySQL value (to read its type and decimal
                places).
            epsilon: Absolute tolerance near zero.

        Returns:
            True if the two numbers match.
        """
        if cls._is_whole(src_num) and cls._is_whole(dest_num):
            return int(src_num) == int(dest_num)
        f1, f2 = float(src_num), float(dest_num)
        if math.isnan(f1) or math.isnan(f2):
            return math.isnan(f1) and math.isnan(f2)
        rel = (
            _MYSQL_FLOAT_REL_TOL
            if isinstance(src_raw, float)
            else _REL_TOL
        )
        if math.isclose(f1, f2, rel_tol=rel, abs_tol=epsilon):
            return True
        places = cls._decimal_places(src_raw)
        if places == 0:
            return False
        try:
            src_dec = decimal.Decimal(str(src_num))
            dest_dec = decimal.Decimal(str(dest_num)).quantize(
                decimal.Decimal(1).scaleb(-places),
                rounding=decimal.ROUND_HALF_UP,
            )
        except (decimal.InvalidOperation, ValueError):
            return False
        return src_dec == dest_dec

    def _values_equal(
        self,
        val1: Any,
        val2: Any,
        epsilon: float = 1e-9,
    ) -> bool:
        """Check whether a MySQL cell value matches a PostgreSQL cell value.

        Rules:
          - Two NULLs match; a NULL and a value do not.
          - Numbers follow `_numbers_match`: whole numbers (counts, sums of
            integers, IDs) exactly; others to one part in a billion (one part
            in a million for MySQL floating-point values), or equal after
            rounding the PostgreSQL value the way MySQL rounded its own.
          - Text must match exactly, including letter case and spaces.

        Args:
            val1: Cell value from the MySQL result (order matters: only the
                MySQL value may be the rounded one).
            val2: Cell value from the PostgreSQL result.
            epsilon: Absolute tolerance for non-whole numbers near zero.

        Returns:
            True if the two values match.
        """
        if val1 is None or val2 is None:
            return val1 is None and val2 is None
        if val1 == val2 and type(val1) is type(val2):
            return True

        num1, num2 = self._as_number(val1), self._as_number(val2)
        if num1 is not None and num2 is not None:
            return self._numbers_match(num1, num2, val1, epsilon)

        # Everything else (text, dates, times) must print identically.
        return str(val1) == str(val2)

    @classmethod
    def _canonicalize_value(
        cls, val: Any, round_digits: Optional[int] = 3
    ) -> Tuple:
        """Turn a cell value into a sortable `(type_tag, ...)` tuple.

        Only used to line rows up before `_values_equal` compares them, so it
        just needs to put equal values next to each other.

        Example:
            - `_canonicalize_value(None)`      -> `("N", 0.0, 0)`
            - `_canonicalize_value("11.3333")` -> `("F", 11.333, 0)`
            - `_canonicalize_value("Biology")` -> `("S", "Biology")`

        Args:
            val: Raw cell value from a database query result.
            round_digits: Number of decimal places to round numbers to when
                sorting rows, so numbers differing only far after the decimal
                point sort into the same place.

        Returns:
            A tuple that Python can sort even when a column mixes NULLs,
            numbers and text.
        """
        if val is None:
            return ("N", 0.0, 0)
        num = cls._as_number(val)
        if num is not None:
            f = float(num)
            if math.isnan(f):
                return ("N", 1.0, 0)
            # Whole numbers also keep their exact value, so two large IDs that
            # round to the same float still sort apart.
            exact = int(num) if cls._is_whole(num) else 0
            rounded = f if round_digits is None else round(f, round_digits)
            return ("F", rounded, exact)
        return ("S", str(val))

    def _rows_match_on_columns(
        self,
        src_rows: Sequence[Tuple],
        dest_rows: Sequence[Tuple],
        cols: Sequence[int],
    ) -> bool:
        """Check whether sorting both tables on `cols` produces identical rows.

        Args:
            src_rows: Rows from the MySQL result.
            dest_rows: Rows from the PostgreSQL result.
            cols: Column indices to sort and compare (all columns in the table).

        Returns:
            True if every row in `src_rows` matches `dest_rows` across `cols` after sorting.
        """
        if not cols or len(src_rows) != len(dest_rows):
            return False

        # Try rounding sort keys to 4, 3, and 2 decimal places when sorting so
        # two numbers like 1.23454 and 1.23456 don't accidentally sort in opposite
        # orders, then verify every cell pair using `_values_equal` (1e-4).
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
