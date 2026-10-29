"""Unit tests for MySQL-to-PostgreSQL query converter and tie-breaker checking.

This module contains:
  - `TestTieAwareResultComparison`: Verifies `MySQLToPGQueryConverter.compare_results`
    (Check 1: exact row-by-row match with 4-decimal rounding tolerance; Check 2:
    same exact rows across 100% of columns when sorted, with zero columns ever
    ignored or dropped so any mismatch in `AVG`, `ROW_NUMBER`, `DENSE_RANK`, or
    running totals is rejected).
  - `TestOrchestratorTieEquivalenceDefault`: Verifies that
    `QueryConversionOrchestrator` defaults `allow_tie_equivalence` to `False`
    for all existing converters while running shared tie-breaker verification
    across both queries for `MySQLToPGQueryConverter`.
  - `TestDeterministicTiebreakerInjection`: Verifies that
    `_inject_deterministic_tiebreakers` adds shared tie-breaker columns to
    `ROW_NUMBER`, `LAG`, moving-window (`ROWS BETWEEN`), and `ORDER BY ... LIMIT`
    queries while leaving `DENSE_RANK` untouched.
"""

import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from db_convertor.query_conversion_orchestrator import (  # pylint: disable=g-import-not-at-top
    QueryConversionOrchestrator,
    _inject_deterministic_tiebreakers,
    _inject_window_tiebreakers,
)
from db_convertor.query_converters.base import (  # pylint: disable=g-import-not-at-top
    ConversionStatus,
    QueryResult,
)
from db_convertor.query_converters.mysql_to_pg import (  # pylint: disable=g-import-not-at-top
    MySQLToPGQueryConverter,
)
from db_convertor.query_converters.sqlite_to_pg import (  # pylint: disable=g-import-not-at-top
    SQLiteToPGQueryConverter,
)


class TestTieAwareResultComparison(unittest.TestCase):
    """Tests for `MySQLToPGQueryConverter.compare_results` (100% column matching, zero column dropping)."""

    def test_mysql_to_pg_converter_dialects_and_prompt(self):
        """Verify MySQLToPGQueryConverter initializes dialects and builds a prompt."""
        converter = MySQLToPGQueryConverter()
        self.assertEqual(converter.source_dialect, "mysql")
        self.assertEqual(converter.dest_dialect, "postgresql")
        self.assertTrue(converter.allow_tie_equivalence)
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
        """Verify two tables with the exact same rows across all columns in different tie orders match."""
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

    def test_no_column_dropping_rejects_any_single_column_mismatch(self):
        """Verify compare_results never drops columns and rejects any single-column mismatch."""
        converter = MySQLToPGQueryConverter()
        src_avg = QueryResult(
            columns=["dept", "cnt", "avg_half"],
            rows=[("EECS", 4, 5.5), ("Physics", 3, 4.5)],
            total_rows=2,
        )
        dest_avg = QueryResult(
            columns=["dept", "cnt", "avg_half"],
            rows=[("EECS", 4, 5.0), ("Physics", 3, 4.0)],
            total_rows=2,
        )
        self.assertFalse(converter.compare_results(src_avg, dest_avg)[0])

    def test_avg_mismatch_alongside_row_number_is_rejected(self):
        """Verify an AVG integer-division bug alongside a ROW_NUMBER column is rejected."""
        converter = MySQLToPGQueryConverter()
        src = QueryResult(
            columns=["dept", "avg_val", "rn"],
            rows=[("EECS", 5.5, 1), ("Physics", 4.5, 2)],
            total_rows=2,
        )
        dest = QueryResult(
            columns=["dept", "avg_val", "rn"],
            rows=[("EECS", 5.0, 1), ("Physics", 4.0, 2)],
            total_rows=2,
        )
        self.assertFalse(converter.compare_results(src, dest)[0])

    def test_unresolved_window_column_mismatch_is_rejected_by_compare_results(self):
        """Verify compare_results rejects an unresolved ROW_NUMBER difference unless resolved by tie-breakers."""
        converter = MySQLToPGQueryConverter()
        src_win = QueryResult(
            columns=["dept", "course", "rn"],
            rows=[("EECS", "6.001", 1), ("EECS", "6.002", 2)],
            total_rows=2,
        )
        dest_win = QueryResult(
            columns=["dept", "course", "rn"],
            rows=[("EECS", "6.002", 1), ("EECS", "6.001", 2)],
            total_rows=2,
        )
        self.assertFalse(converter.compare_results(src_win, dest_win)[0])

    def test_dense_rank_collation_mismatch_is_rejected(self):
        """Verify a DENSE_RANK difference caused by sort-order/collation differences is rejected."""
        converter = MySQLToPGQueryConverter()
        src = QueryResult(
            columns=["code", "dr"],
            rows=[("A-1", 1), ("A 2", 2), ("A-3", 3)],
            total_rows=3,
        )
        dest = QueryResult(
            columns=["code", "dr"],
            rows=[("A 2", 1), ("A-1", 2), ("A-3", 3)],
            total_rows=3,
        )
        self.assertFalse(converter.compare_results(src, dest)[0])

    def test_unframed_running_sum_mismatch_is_rejected(self):
        """Verify a running-total SUM OVER (ORDER BY ...) difference is rejected."""
        converter = MySQLToPGQueryConverter()
        src = QueryResult(
            columns=["code", "running_total"],
            rows=[("A", 10), ("B", 30)],
            total_rows=2,
        )
        dest = QueryResult(
            columns=["code", "running_total"],
            rows=[("A", 20), ("B", 30)],
            total_rows=2,
        )
        self.assertFalse(converter.compare_results(src, dest)[0])

    def test_top1_sort_key_aggregate_mismatch_is_rejected(self):
        """Verify a top-1 query where the sort-key aggregate (e.g. COUNT) differs is rejected."""
        converter = MySQLToPGQueryConverter()
        src = QueryResult(
            columns=["dept", "top_course", "student_count"],
            rows=[("EECS", "6.001", 120)],
            total_rows=1,
        )
        dest = QueryResult(
            columns=["dept", "top_course", "student_count"],
            rows=[("EECS", "6.002", 95)],
            total_rows=1,
        )
        self.assertFalse(converter.compare_results(src, dest)[0])

    def test_genuine_base_row_mismatch_is_rejected(self):
        """Verify queries returning different rows fail."""
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


class TestOrchestratorTieEquivalenceDefault(unittest.TestCase):
    """Tests that `QueryConversionOrchestrator` defaults tie equivalence to `False`."""

    def test_orchestrator_defaults_tie_equivalence_to_false_for_existing_converters(self):
        """Verify existing converters without `allow_tie_equivalence` stay strict and skip tie-breaker retries."""
        sqlite_converter = SQLiteToPGQueryConverter()
        self.assertFalse(hasattr(sqlite_converter, "allow_tie_equivalence"))

        orchestrator = QueryConversionOrchestrator(
            converter=sqlite_converter,
            source_connection={"path": ":memory:"},
            dest_connection={},
            source_schema="CREATE TABLE t (grp TEXT, v INT);",
            dest_schema='CREATE TABLE "t" ("grp" TEXT, "v" INT);',
            max_attempts=1,
        )
        source_sql = (
            "SELECT grp, v, ROW_NUMBER() OVER (ORDER BY grp) AS rn FROM t"
        )
        candidate_pg_sql = (
            'SELECT "grp", "v", ROW_NUMBER() OVER (ORDER BY "grp") AS "rn" FROM "t"'
        )

        src_res = QueryResult(
            columns=["grp", "v", "rn"],
            rows=[("A", 10, 1), ("A", 20, 2)],
            total_rows=2,
        )
        dest_res = QueryResult(
            columns=["grp", "v", "rn"],
            rows=[("A", 20, 1), ("A", 10, 2)],
            total_rows=2,
        )
        llm_reply = json.dumps({
            "converted_query": candidate_pg_sql,
            "notes": "Initial conversion",
        })

        with (
            mock.patch.object(
                orchestrator, "_execute_source_query", return_value=src_res
            ),
            mock.patch.object(
                orchestrator, "_execute_dest_query", return_value=dest_res
            ),
            mock.patch.object(
                orchestrator, "_check_shared_window_tiebreaker"
            ) as mock_tb,
            mock.patch(
                "db_convertor.query_conversion_orchestrator.gemini_inference",
                return_value=llm_reply,
            ),
        ):
            result = orchestrator.convert_query(
                source_sql, question_id="q1", verbose=False
            )
            mock_tb.assert_not_called()
            self.assertEqual(result.status, ConversionStatus.EXHAUSTED_RETRY)

    def test_orchestrator_resolves_tie_via_deterministic_tiebreaker_on_both_queries(self):
        """Verify MySQLToPGQueryConverter runs shared deterministic tie-breakers and requires 100%-column match."""
        mysql_converter = MySQLToPGQueryConverter()
        orchestrator = QueryConversionOrchestrator(
            converter=mysql_converter,
            source_connection={},
            dest_connection={},
            source_schema="CREATE TABLE t (id INT, grp VARCHAR(10), v INT);",
            dest_schema='CREATE TABLE "t" ("id" INT, "grp" VARCHAR(10), "v" INT);',
            max_attempts=1,
        )
        source_sql = (
            "SELECT t.grp, t.v, ROW_NUMBER() OVER (ORDER BY t.grp) AS rn FROM t"
        )
        candidate_pg_sql = (
            'SELECT "t"."grp", "t"."v", ROW_NUMBER() OVER (ORDER BY "t"."grp") AS "rn" FROM "t"'
        )
        # Before tie-breaking, tied rows in grp='A' get opposite ROW_NUMBER() values.
        src_initial = QueryResult(
            columns=["grp", "v", "rn"],
            rows=[("A", 10, 1), ("A", 20, 2)],
            total_rows=2,
        )
        dest_initial = QueryResult(
            columns=["grp", "v", "rn"],
            rows=[("A", 20, 1), ("A", 10, 2)],
            total_rows=2,
        )
        # After deterministic tie-breaking on (t.grp, t.v, t.id), both queries produce identical rows across all 3 columns.
        tiebroken_aligned = QueryResult(
            columns=["grp", "v", "rn"],
            rows=[("A", 10, 1), ("A", 20, 2)],
            total_rows=2,
        )
        llm_reply = json.dumps({
            "converted_query": candidate_pg_sql,
            "notes": "Initial conversion",
        })

        with (
            mock.patch.object(
                orchestrator,
                "_execute_source_query",
                side_effect=[src_initial, tiebroken_aligned],
            ) as mock_src_exec,
            mock.patch.object(
                orchestrator,
                "_execute_dest_query",
                side_effect=[dest_initial, tiebroken_aligned],
            ) as mock_dest_exec,
            mock.patch(
                "db_convertor.query_conversion_orchestrator.gemini_inference",
                return_value=llm_reply,
            ),
        ):
            result = orchestrator.convert_query(
                source_sql, question_id="q2", verbose=False
            )
            self.assertEqual(result.status, ConversionStatus.RESULT_MATCHED)
            self.assertEqual(mock_src_exec.call_count, 2)
            self.assertEqual(mock_dest_exec.call_count, 2)
            self.assertIn("shared window tie-breaker", result.reason)


class TestDeterministicTiebreakerInjection(unittest.TestCase):
    """Tests for `_inject_deterministic_tiebreakers` and `_inject_window_tiebreakers`."""

    def test_injects_other_select_columns_and_agg_arg_into_rows_between(self):
        """Verify the other SELECT columns and window input column are added to ORDER BY."""
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

    def test_leaves_dense_rank_untouched_while_breaking_ties_on_lag(self):
        """Verify DENSE_RANK() is left untouched (so tied rows keep the same rank) while LAG() gets tie-breakers."""
        sql = (
            "SELECT dept, course, "
            "DENSE_RANK() OVER (ORDER BY dept) AS dr, "
            "LAG(course) OVER (ORDER BY dept) AS prev_course "
            "FROM courses"
        )
        rewritten = _inject_deterministic_tiebreakers(sql)
        # DENSE_RANK() OVER (ORDER BY dept) must be left untouched so tied rows share the same rank.
        self.assertIn("DENSE_RANK() OVER (ORDER BY dept)", rewritten)
        # LAG(course) OVER (ORDER BY dept) must get the tie-breaker columns added.
        self.assertIn(
            "LAG(course) OVER (ORDER BY dept, dept, course, course)",
            rewritten,
        )

    def test_injects_tiebreakers_into_order_by_limit(self):
        """Verify ORDER BY ... LIMIT queries receive deterministic tie-breakers from SELECT."""
        sql = (
            "SELECT s.department_code, s.course_title "
            "FROM courses s "
            "ORDER BY s.department_code ASC LIMIT 5"
        )
        rewritten = _inject_deterministic_tiebreakers(sql, dialect="mysql")
        self.assertIn(
            "s.course_title IS NOT NULL ASC, s.course_title ASC LIMIT 5",
            rewritten,
        )

    def test_postgresql_case_insensitive_upper_tiebreaker_alignment(self):
        """Verify PostgreSQL tie-breaker injection adds UPPER(...) + column for text columns."""
        schema_ddl = 'CREATE TABLE "courses" ("id" INT, "dept" VARCHAR(20), "title" VARCHAR(100));'
        sql = (
            'SELECT c."dept", c."title", '
            'ROW_NUMBER() OVER (ORDER BY c."dept") AS rn '
            'FROM "courses" c'
        )
        rewritten = _inject_deterministic_tiebreakers(
            sql, dialect="postgresql", schema_ddl=schema_ddl
        )
        self.assertIn(
            'UPPER(REPLACE(c."title", \'-\', \' \')) COLLATE "C" ASC NULLS FIRST',
            rewritten,
        )
        self.assertIn('"c"."id" ASC NULLS FIRST', rewritten)


if __name__ == "__main__":
    unittest.main()
