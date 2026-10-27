"""Unit tests for MySQL-to-PostgreSQL query converter and tie-aware result checking.

This module contains:
  - `TestTieAwareResultComparison`: Verifies the mechanical tie-equivalence
    checks in `QueryConverter.compare_results` (strict positional with 4-decimal
    rounding tolerance, order-independent multiset row match, moving-window /
    top-1 tie equivalence, ranking/lag tie equivalence, and strict mode).
  - `TestSharedWindowTiebreakerInjection`: Verifies `_inject_window_tiebreakers`
    for sliding-window queries (`ROWS BETWEEN ...`) with outer filters or
    `SELECT DISTINCT`.
"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from db_convertor.query_conversion_orchestrator import _inject_window_tiebreakers  # pylint: disable=g-import-not-at-top
from db_convertor.query_converters.base import QueryResult  # pylint: disable=g-import-not-at-top
from db_convertor.query_converters.mysql_to_pg import MySQLToPGQueryConverter  # pylint: disable=g-import-not-at-top


class TestTieAwareResultComparison(unittest.TestCase):
    """Tests for `QueryConverter.compare_results` with and without tie equivalence."""

    def test_mysql_to_pg_converter_dialects_and_prompt(self):
        """Verify MySQLToPGQueryConverter initializes dialects and builds a prompt."""
        converter = MySQLToPGQueryConverter()
        self.assertEqual(converter.source_dialect, "mysql")
        self.assertEqual(converter.dest_dialect, "postgresql")
        prompt = converter.get_conversion_prompt(
            source_schema="CREATE TABLE t (id INT);",
            dest_schema='CREATE TABLE "t" ("id" INTEGER);',
            source_query="SELECT `id` FROM `t`;",
        )
        self.assertIn("SOURCE (MySQL):", prompt)
        self.assertIn("DESTINATION (PostgreSQL):", prompt)

    def test_strict_positional_allows_four_decimal_truncation(self):
        """Verify 4-decimal MySQL AVG/DIV truncation matches full float precision."""
        converter = MySQLToPGQueryConverter()
        src = QueryResult(
            columns=["dept", "avg_units"],
            rows=[("Physics", "11.3333"), ("Math", "9.6667")],
            total_rows=2,
        )
        dest = QueryResult(
            columns=["dept", "avg_units"],
            rows=[("Physics", 11.333333333333334), ("Math", 9.666666666666666)],
            total_rows=2,
        )
        matches, reason = converter.compare_results(src, dest)
        self.assertTrue(matches)
        self.assertIn("data matches", reason)

    def test_same_rows_reordered_on_ties(self):
        """Verify identical row multisets returned in different tie orders match."""
        converter = MySQLToPGQueryConverter()
        src = QueryResult(
            columns=["term_code", "subject_id", "units"],
            rows=[
                ("2024FA", "18.01", 12),
                ("2024FA", "18.02", 12),
                ("2024FA", "18.03", 12),
            ],
            total_rows=3,
        )
        dest = QueryResult(
            columns=["term_code", "subject_id", "units"],
            rows=[
                ("2024FA", "18.03", 12),
                ("2024FA", "18.01", 12),
                ("2024FA", "18.02", 12),
            ],
            total_rows=3,
        )
        matches, reason = converter.compare_results(src, dest)
        self.assertTrue(matches)
        self.assertIn("same rows reordered", reason)

        # When allow_tie_equivalence is False, reordered rows must be rejected.
        strict_converter = MySQLToPGQueryConverter(allow_tie_equivalence=False)
        strict_matches, _ = strict_converter.compare_results(src, dest)
        self.assertFalse(strict_matches)

    def test_moving_window_tie_equivalence(self):
        """Verify moving-window column differences on tied sort keys pass when base rows match."""
        converter = MySQLToPGQueryConverter()
        src = QueryResult(
            columns=["subject_title", "term_start", "duration", "moving_avg"],
            rows=[
                ("Intro CS", "20240904", 90, 95.0),
                ("Algorithms", "20240904", 100, 96.0),
                ("Systems", "20240904", 98, 94.0),
            ],
            total_rows=3,
        )
        dest = QueryResult(
            columns=["subject_title", "term_start", "duration", "moving_avg"],
            rows=[
                ("Algorithms", "20240904", 100, 94.0),
                ("Systems", "20240904", 98, 96.0),
                ("Intro CS", "20240904", 90, 94.5),
            ],
            total_rows=3,
        )
        matches, reason = converter.compare_results(src, dest)
        self.assertTrue(matches)
        self.assertIn("tie-equivalent", reason)

    def test_row_number_permutation_on_ties(self):
        """Verify ROW_NUMBER() assigned in different tie orders passes when base rows match."""
        converter = MySQLToPGQueryConverter()
        src = QueryResult(
            columns=["dept", "course", "rn"],
            rows=[
                ("EECS", "6.001", 1),
                ("EECS", "6.002", 2),
                ("EECS", "6.003", 3),
            ],
            total_rows=3,
        )
        dest = QueryResult(
            columns=["dept", "course", "rn"],
            rows=[
                ("EECS", "6.003", 1),
                ("EECS", "6.001", 2),
                ("EECS", "6.002", 3),
            ],
            total_rows=3,
        )
        matches, reason = converter.compare_results(src, dest)
        self.assertTrue(matches)
        self.assertIn("tie-equivalent", reason)

    def test_genuine_base_row_mismatch_is_rejected(self):
        """Verify queries returning genuinely different underlying base rows fail."""
        converter = MySQLToPGQueryConverter()
        src = QueryResult(
            columns=["dept", "course", "moving_avg"],
            rows=[("EECS", "6.001", 10.0), ("EECS", "6.002", 12.0)],
            total_rows=2,
        )
        dest = QueryResult(
            columns=["dept", "course", "moving_avg"],
            rows=[("EECS", "6.001", 10.0), ("Physics", "8.01", 12.0)],
            total_rows=2,
        )
        matches, _ = converter.compare_results(src, dest)
        self.assertFalse(matches)


class TestSharedWindowTiebreakerInjection(unittest.TestCase):
    """Tests for `_inject_window_tiebreakers` on sliding-window queries."""

    def test_injects_sibling_columns_and_agg_arg_into_rows_between(self):
        """Verify sibling SELECT expressions and agg arg are appended to ORDER BY."""
        sql = (
            "SELECT s.subject_title, s.department_name, "
            "MIN(s.total_units) OVER (ORDER BY at.academic_year ASC NULLS LAST "
            "ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING) AS window_column "
            "FROM subject_summary s"
        )
        rewritten = _inject_window_tiebreakers(sql)
        self.assertIn(
            "ORDER BY at.academic_year, s.subject_title, s.department_name, s.total_units ROWS BETWEEN",
            rewritten,
        )


if __name__ == "__main__":
    unittest.main()
