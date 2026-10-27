"""MySQL to PostgreSQL SQL query converter with tie-aware mechanical checking.

This module contains:
  - `MySQLToPGQueryConverter`: Builds the ReAct prompt for converting MySQL
    queries to PostgreSQL and implements `compare_results` with mechanical
    tie-equivalence checks when queries contain untied `ORDER BY` or window
    clauses.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .base import QueryConverter, QueryResult


class MySQLToPGQueryConverter(QueryConverter):
    """Convert SQL queries from MySQL to PostgreSQL dialect."""

    def __init__(self, allow_tie_equivalence: bool = True):
        """Initialize the MySQL to PostgreSQL query converter.

        Args:
            allow_tie_equivalence: Whether `compare_results` should accept
                tie-equivalent row ordering and window results after running the
                strict row-by-row check. Defaults to True.
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
                            source_result, dest_result
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
    ) -> Tuple[bool, str]:
        """Compare MySQL and PostgreSQL query results using mechanical checks.

        Checks in order:
          1. Original row-by-row check in the returned order (allowing `1e-4`
             decimal rounding tolerance).
          2. Reordered rows on ties: If row and column counts match, checks if
             sorting both tables produces the exact same rows across all columns.
          3. Moving-window or rank ties: If row and column counts match, checks
             if all underlying non-window columns match 100% when 1 to 3
             tie-dependent window/ranking columns are set aside.

        Args:
            source_result: Result from executing the MySQL query.
            dest_result: Result from executing the PostgreSQL query.

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

        # 2. Check if sorting both tables produces the exact same rows across all columns.
        if self._rows_match_on_columns(
            source_result.rows, dest_result.rows, all_cols
        ):
            return (
                True,
                "All checks passed (same rows reordered): exact same set of rows "
                "across all columns (only row order differs on ties)",
            )

        # 3. Check if setting aside 1 to 3 tie-dependent window/rank columns leaves
        # all remaining base columns matching 100% as a set of rows.
        if ncols >= 2:
            matching_cols = []
            diff_cols = []
            for col_idx in all_cols:
                if self._column_values_match(
                    source_result.rows, dest_result.rows, col_idx
                ):
                    matching_cols.append(col_idx)
                else:
                    diff_cols.append(col_idx)

            min_base_cols = 1 if ncols == 2 else max(2, (ncols + 1) // 2)
            if (
                len(matching_cols) >= min_base_cols
                and 1 <= len(diff_cols) <= 3
                and self._rows_match_on_columns(
                    source_result.rows, dest_result.rows, matching_cols
                )
            ):
                diff_names = [dest_result.columns[c] for c in diff_cols]
                return (
                    True,
                    f"All checks passed (window/top-1 tie-equivalent): all "
                    f"{len(matching_cols)}/{ncols} base columns match 100%; "
                    f"only tie-dependent column(s) {diff_names} differ",
                )

            max_drop = min(3, ncols - min_base_cols)
            for num_drop in range(1, max_drop + 1):
                for dropped in self._column_combinations(all_cols, num_drop):
                    if not set(diff_cols).issubset(dropped):
                        continue
                    rem_cols = [c for c in all_cols if c not in dropped]
                    if len(rem_cols) < min_base_cols:
                        continue
                    if self._rows_match_on_columns(
                        source_result.rows, dest_result.rows, rem_cols
                    ):
                        dropped_names = [dest_result.columns[c] for c in dropped]
                        return (
                            True,
                            f"All checks passed (window/rank tie-equivalent): all "
                            f"{len(rem_cols)}/{ncols} base columns match 100%; "
                            f"only tie-dependent column(s) {dropped_names} differ",
                        )

        return False, first_mismatch_reason

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
    def _canonicalize_value(val: Any) -> Tuple[str, Any]:
        """Convert a cell value into a sortable key tuple for row comparison.

        Args:
            val: Raw cell value from a database driver.

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
            return ("F", round(num, 3))
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
        src_canon = sorted(self._canonicalize_value(r[col_idx]) for r in src_rows)
        dest_canon = sorted(self._canonicalize_value(r[col_idx]) for r in dest_rows)
        if src_canon == dest_canon:
            return True

        src_sorted = [
            r[col_idx]
            for _, r in sorted(
                zip(
                    [self._canonicalize_value(r[col_idx]) for r in src_rows],
                    src_rows,
                ),
                key=lambda item: item[0],
            )
        ]
        dest_sorted = [
            r[col_idx]
            for _, r in sorted(
                zip(
                    [self._canonicalize_value(r[col_idx]) for r in dest_rows],
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

        Args:
            src_rows: Rows from the source database.
            dest_rows: Rows from the destination database.
            cols: Sequence of column indices to compare.

        Returns:
            True if every row in `src_rows` matches `dest_rows` on `cols` after sorting.
        """
        if not cols:
            return False

        def row_sort_key(row: Tuple) -> Tuple[Tuple[str, Any], ...]:
            """Build a canonical sort key for a row over the selected columns."""
            return tuple(self._canonicalize_value(row[c]) for c in cols)

        src_canon = sorted(row_sort_key(r) for r in src_rows)
        dest_canon = sorted(row_sort_key(r) for r in dest_rows)
        if src_canon == dest_canon:
            return True

        src_sorted = [
            tuple(r[c] for c in cols)
            for _, r in sorted(
                zip([row_sort_key(r) for r in src_rows], src_rows),
                key=lambda item: item[0],
            )
        ]
        dest_sorted = [
            tuple(r[c] for c in cols)
            for _, r in sorted(
                zip([row_sort_key(r) for r in dest_rows], dest_rows),
                key=lambda item: item[0],
            )
        ]
        return all(
            all(self._values_equal(a, b) for a, b in zip(src_r, dest_r))
            for src_r, dest_r in zip(src_sorted, dest_sorted)
        )

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
