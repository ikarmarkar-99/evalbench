"""MySQL to PostgreSQL SQL query converter with tie-aware result comparison.

This module contains:
  - `MySQLToPGQueryConverter`: Builds the prompt for converting MySQL queries to
    PostgreSQL and implements `compare_results` to compare two result tables
    across 100% of columns:
      1. Check 1: Exact row-by-row match in the returned order (allowing 4-decimal
         rounding tolerance between MySQL and PostgreSQL).
      2. Check 2: Same exact rows across all columns when sorted (so rows that
         only swapped positions because of a tie still pass, while zero columns
         are ever ignored or dropped).
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .base import QueryConverter, QueryResult


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
        """Compare MySQL and PostgreSQL query results across 100% of columns.

        Checks in order:
          1. Check 1 (exact row order): Compares row 1 to row 1, row 2 to row 2,
             and so on across every column, allowing 4-decimal rounding tolerance
             (`1e-4`) because MySQL often rounds `AVG` and division to 4 decimals.
          2. Check 2 (same rows in a different tie order): If `allow_tie_equivalence`
             is True, sorts the rows of both tables across all columns and checks
             whether every row and every column is 100% identical. Zero columns
             are ever ignored or dropped.

        Args:
            source_result: Result from running the MySQL query.
            dest_result: Result from running the PostgreSQL query.

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

        ncols = len(source_result.columns)
        all_cols = list(range(ncols))

        # Check 2: Sort both tables across all columns to see if they contain
        # the exact same rows in a different tie order.
        if self._rows_match_on_columns(
            source_result.rows, dest_result.rows, all_cols
        ):
            return (
                True,
                "All checks passed (same rows reordered): exact same set of rows "
                "across all columns (only row order differs on ties)",
            )

        return False, first_mismatch_reason

    def _values_equal(
        self,
        val1: Any,
        val2: Any,
        rel_tol: float = 1e-4,
        abs_tol: float = 1e-4,
    ) -> bool:
        """Check whether two cell values match (allowing 4-decimal rounding tolerance).

        Args:
            val1: Cell value from the MySQL result.
            val2: Cell value from the PostgreSQL result.
            rel_tol: Relative tolerance for comparing decimal numbers.
            abs_tol: Absolute tolerance for comparing decimal numbers.

        Returns:
            True if the two values match.
        """
        # Two empty (NULL) cells match; one NULL and one non-NULL do not.
        if val1 is None and val2 is None:
            return True
        if val1 is None or val2 is None:
            return False

        str1 = str(val1).strip()
        str2 = str(val2).strip()

        # Case-insensitive string match (MySQL default text comparison is case-insensitive).
        if str1.lower() == str2.lower():
            return True

        # If both values are numbers, compare them within 0.0001 tolerance.
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
        """Turn a cell value into a normalized `(type_tag, value)` pair for sorting rows.

        Example:
            - `_canonicalize_value(None)`      -> `("N", "")`
            - `_canonicalize_value("11.3333")` -> `("F", 11.333)`
            - `_canonicalize_value("Biology")` -> `("S", "biology")`

        Args:
            val: Raw cell value from a database query result.
            round_digits: Number of decimal places to round numbers to when
                sorting rows so that numbers differing only at the 5th decimal
                place sort into the same bucket.

        Returns:
            Tuple of `(type_tag, normalized_value)` that Python can sort cleanly
            without crashing when a column mixes NULLs, numbers, and strings.
        """
        # Tag NULLs with "N", numbers with "F", and strings with "S" so Python
        # never tries to compare None < float or float < str directly.
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
        if not cols:
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
