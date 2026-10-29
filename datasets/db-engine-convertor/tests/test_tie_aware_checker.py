"""Unit tests for MySQL-to-PostgreSQL query converter and tie-aware result checking.

This module contains:
  - `TestTieAwareResultComparison`: Verifies the mechanical tie-equivalence
    checks in `MySQLToPGQueryConverter.compare_results` (strict positional with
    4-decimal rounding tolerance, order-independent multiset row match, verified
    physical-order window/top-1 SQL provenance, rejection of integer-division
    `AVG` bugs and peer-invariant `DENSE_RANK`/`RANGE` collation bugs, and
    strict mode).
  - `TestOrchestratorTieEquivalenceDefault`: Verifies that
    `QueryConversionOrchestrator` defaults `allow_tie_equivalence` to `False`
    for all existing converters while enabling tie-equivalence and shared
    `ROWS BETWEEN` tie-breaker retries for `MySQLToPGQueryConverter`.
  - `TestSharedWindowTiebreakerInjection`: Verifies `_inject_window_tiebreakers`
    for sliding-window queries (`ROWS BETWEEN ...`) with outer filters or
    `SELECT DISTINCT`.
"""

import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from db_convertor.query_conversion_orchestrator import (  # pylint: disable=g-import-not-at-top
    QueryConversionOrchestrator,
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
    """Tests for `MySQLToPGQueryConverter.compare_results` and SQL provenance guards."""

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
        """Verify verified ROWS BETWEEN window differences on tied keys pass when base rows match."""
        converter = MySQLToPGQueryConverter()
        src_sql = (
            "SELECT subject_title, term_start, duration, "
            "AVG(duration) OVER (ORDER BY term_start "
            "ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS moving_avg "
            "FROM courses"
        )
        dest_sql = (
            'SELECT "subject_title", "term_start", "duration", '
            'AVG("duration") OVER (ORDER BY "term_start" ASC NULLS FIRST '
            'ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS "moving_avg" '
            'FROM "courses"'
        )
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
        matches, reason = converter.compare_results(
            src, dest, source_query=src_sql, converted_query=dest_sql
        )
        self.assertTrue(matches)
        self.assertIn("tie-equivalent", reason)

    def test_row_number_permutation_on_ties(self):
        """Verify ROW_NUMBER() assigned in different tie orders passes when base rows match."""
        converter = MySQLToPGQueryConverter()
        src_sql = (
            "SELECT dept, course, ROW_NUMBER() OVER (ORDER BY dept) AS rn "
            "FROM courses"
        )
        dest_sql = (
            'SELECT "dept", "course", '
            'ROW_NUMBER() OVER (ORDER BY "dept" ASC NULLS FIRST) AS "rn" '
            'FROM "courses"'
        )
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
        matches, reason = converter.compare_results(
            src, dest, source_query=src_sql, converted_query=dest_sql
        )
        self.assertTrue(matches)
        self.assertIn("tie-equivalent", reason)

    def test_check3_requires_sql_queries(self):
        """Verify Check 3 never drops columns when SQL queries are omitted."""
        converter = MySQLToPGQueryConverter()
        src = QueryResult(
            columns=["dept", "course", "moving_avg"],
            rows=[
                ("EECS", "6.001", 95.0),
                ("EECS", "6.002", 96.0),
            ],
            total_rows=2,
        )
        dest = QueryResult(
            columns=["dept", "course", "moving_avg"],
            rows=[
                ("EECS", "6.002", 94.0),
                ("EECS", "6.001", 94.5),
            ],
            total_rows=2,
        )
        # Without source_query/converted_query, Check 3 cannot verify column provenance.
        matches, _ = converter.compare_results(src, dest)
        self.assertFalse(matches)

    def test_integer_division_in_avg_column_is_rejected(self):
        """Verify integer division bugs in AVG columns fail even alongside ROW_NUMBER."""
        converter = MySQLToPGQueryConverter()
        # Case A: Plain GROUP BY query with AVG(units / 2) integer truncation in PG.
        src_sql_plain = (
            "SELECT dept, COUNT(*) AS cnt, AVG(units / 2) AS avg_half "
            "FROM courses GROUP BY dept"
        )
        dest_sql_plain = (
            'SELECT "dept", COUNT(*) AS "cnt", AVG("units" / 2) AS "avg_half" '
            'FROM "courses" GROUP BY "dept"'
        )
        src_plain = QueryResult(
            columns=["dept", "cnt", "avg_half"],
            rows=[("EECS", 4, 5.5), ("Physics", 3, 4.5)],
            total_rows=2,
        )
        dest_plain = QueryResult(
            columns=["dept", "cnt", "avg_half"],
            rows=[("EECS", 4, 5.0), ("Physics", 3, 4.0)],
            total_rows=2,
        )
        matches_plain, _ = converter.compare_results(
            src_plain,
            dest_plain,
            source_query=src_sql_plain,
            converted_query=dest_sql_plain,
        )
        self.assertFalse(matches_plain)

        # Case B: AVG(units / 2) bug sitting alongside a genuine ROW_NUMBER() column.
        src_sql_mixed = (
            "SELECT dept, COUNT(*) AS cnt, AVG(units / 2) AS avg_half, "
            "ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rn "
            "FROM courses GROUP BY dept"
        )
        dest_sql_mixed = (
            'SELECT "dept", COUNT(*) AS "cnt", AVG("units" / 2) AS "avg_half", '
            'ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS "rn" '
            'FROM "courses" GROUP BY "dept"'
        )
        src_mixed = QueryResult(
            columns=["dept", "cnt", "avg_half", "rn"],
            rows=[("EECS", 4, 5.5, 1), ("Physics", 4, 4.5, 2)],
            total_rows=2,
        )
        dest_mixed = QueryResult(
            columns=["dept", "cnt", "avg_half", "rn"],
            rows=[("Physics", 4, 4.0, 1), ("EECS", 4, 5.0, 2)],
            total_rows=2,
        )
        matches_mixed, _ = converter.compare_results(
            src_mixed,
            dest_mixed,
            source_query=src_sql_mixed,
            converted_query=dest_sql_mixed,
        )
        self.assertFalse(matches_mixed)

    def test_integer_division_inside_window_avg_is_rejected(self):
        """Verify uncast PostgreSQL integer division inside a ROWS BETWEEN window is rejected."""
        converter = MySQLToPGQueryConverter()
        src_sql = (
            "SELECT dept, course, "
            "AVG(units / 2) OVER (ORDER BY dept "
            "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS win_avg "
            "FROM courses"
        )
        dest_sql_buggy = (
            'SELECT "dept", "course", '
            'AVG("units" / 2) OVER (ORDER BY "dept" '
            'ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS "win_avg" '
            'FROM "courses"'
        )
        src = QueryResult(
            columns=["dept", "course", "win_avg"],
            rows=[
                ("EECS", "6.001", 4.5),
                ("EECS", "6.002", 5.25),
                ("EECS", "6.003", 5.5),
            ],
            total_rows=3,
        )
        dest_buggy = QueryResult(
            columns=["dept", "course", "win_avg"],
            rows=[
                ("EECS", "6.002", 5.0),
                ("EECS", "6.001", 4.0),
                ("EECS", "6.003", 5.0),
            ],
            total_rows=3,
        )
        matches, _ = converter.compare_results(
            src,
            dest_buggy,
            source_query=src_sql,
            converted_query=dest_sql_buggy,
        )
        self.assertFalse(matches)

    def test_peer_invariant_dense_rank_and_range_sum_mismatch_rejected(self):
        """Verify DENSE_RANK() and unframed RANGE SUM() mismatches fail Check 3."""
        converter = MySQLToPGQueryConverter()
        # DENSE_RANK() assigns identical values to tied peers, so value mismatches
        # indicate a real sort/collation difference rather than arbitrary tie order.
        src_dr_sql = (
            "SELECT subject_id, title, "
            "DENSE_RANK() OVER (ORDER BY subject_id) AS dr FROM subjects"
        )
        dest_dr_sql = (
            'SELECT "subject_id", "title", '
            'DENSE_RANK() OVER (ORDER BY "subject_id") AS "dr" FROM "subjects"'
        )
        src_dr = QueryResult(
            columns=["subject_id", "title", "dr"],
            rows=[("1.054", "Mechanics", 1), ("10.492B", "Transport", 2)],
            total_rows=2,
        )
        dest_dr = QueryResult(
            columns=["subject_id", "title", "dr"],
            rows=[("10.492B", "Transport", 1), ("1.054", "Mechanics", 2)],
            total_rows=2,
        )
        self.assertFalse(
            converter.compare_results(
                src_dr,
                dest_dr,
                source_query=src_dr_sql,
                converted_query=dest_dr_sql,
            )[0]
        )

        # Unframed SUM(...) OVER (ORDER BY ...) uses RANGE UNBOUNDED PRECEDING,
        # which sums full peer groups deterministically regardless of tie order.
        src_sum_sql = (
            "SELECT subject_id, units, "
            "SUM(units) OVER (ORDER BY subject_id) AS running_units FROM subjects"
        )
        dest_sum_sql = (
            'SELECT "subject_id", "units", '
            'SUM("units") OVER (ORDER BY "subject_id") AS "running_units" FROM "subjects"'
        )
        src_sum = QueryResult(
            columns=["subject_id", "units", "running_units"],
            rows=[("1.054", 9, 9), ("10.492B", 12, 21)],
            total_rows=2,
        )
        dest_sum = QueryResult(
            columns=["subject_id", "units", "running_units"],
            rows=[("10.492B", 12, 12), ("1.054", 9, 21)],
            total_rows=2,
        )
        self.assertFalse(
            converter.compare_results(
                src_sum,
                dest_sum,
                source_query=src_sum_sql,
                converted_query=dest_sum_sql,
            )[0]
        )

    def test_cte_and_top1_rank_provenance_accepted_and_sort_key_protected(self):
        """Verify top-1 `rn = 1` CTE payload ties pass while aggregate sort keys stay protected."""
        converter = MySQLToPGQueryConverter()
        src_sql = (
            "WITH ranked AS ("
            "  SELECT dept, course_title, COUNT(*) AS student_cnt, "
            "  ROW_NUMBER() OVER (PARTITION BY dept ORDER BY COUNT(*) DESC) AS rn "
            "  FROM enrollments GROUP BY dept, course_title"
            ") "
            "SELECT dept, course_title, student_cnt FROM ranked WHERE rn = 1"
        )
        dest_sql = (
            'WITH "ranked" AS ('
            '  SELECT "dept", "course_title", COUNT(*) AS "student_cnt", '
            '  ROW_NUMBER() OVER (PARTITION BY "dept" ORDER BY COUNT(*) DESC) AS "rn" '
            '  FROM "enrollments" GROUP BY "dept", "course_title"'
            ") "
            'SELECT "dept", "course_title", "student_cnt" FROM "ranked" WHERE "rn" = 1'
        )
        # Tied top-1 course_title differs for EECS, while dept and student_cnt match 100%.
        src_ok = QueryResult(
            columns=["dept", "course_title", "student_cnt"],
            rows=[("EECS", "Algorithms", 50), ("Physics", "Mechanics", 30)],
            total_rows=2,
        )
        dest_ok = QueryResult(
            columns=["dept", "course_title", "student_cnt"],
            rows=[("EECS", "Systems", 50), ("Physics", "Mechanics", 30)],
            total_rows=2,
        )
        matches, _ = converter.compare_results(
            src_ok, dest_ok, source_query=src_sql, converted_query=dest_sql
        )
        self.assertTrue(matches)

        # If the aggregate sort key `student_cnt` differs, Check 3 MUST reject.
        dest_bad_cnt = QueryResult(
            columns=["dept", "course_title", "student_cnt"],
            rows=[("EECS", "Systems", 49), ("Physics", "Mechanics", 30)],
            total_rows=2,
        )
        matches_bad, _ = converter.compare_results(
            src_ok, dest_bad_cnt, source_query=src_sql, converted_query=dest_sql
        )
        self.assertFalse(matches_bad)

    def test_genuine_base_row_mismatch_is_rejected(self):
        """Verify queries returning genuinely different underlying base rows fail."""
        converter = MySQLToPGQueryConverter()
        src_sql = (
            "SELECT dept, course, "
            "AVG(units) OVER (ORDER BY dept ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS moving_avg "
            "FROM courses"
        )
        dest_sql = (
            'SELECT "dept", "course", '
            'AVG("units") OVER (ORDER BY "dept" ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS "moving_avg" '
            'FROM "courses"'
        )
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
        matches, _ = converter.compare_results(
            src, dest, source_query=src_sql, converted_query=dest_sql
        )
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
            "SELECT DISTINCT grp, MIN(v) OVER (ORDER BY grp "
            "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS w FROM t"
        )
        candidate_pg_sql = (
            'SELECT DISTINCT "grp", MIN("v") OVER (ORDER BY "grp" '
            'ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS "w" FROM "t"'
        )

        # Simulate a row-count mismatch on a ROWS BETWEEN query.
        src_res = QueryResult(
            columns=["grp", "w"], rows=[("A", 1), ("A", 2)], total_rows=2
        )
        dest_res = QueryResult(
            columns=["grp", "w"], rows=[("A", 1)], total_rows=1
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
            # Because SQLiteToPGQueryConverter does not set allow_tie_equivalence=True,
            # the orchestrator must default to False and NEVER call _check_shared_window_tiebreaker.
            mock_tb.assert_not_called()
            self.assertEqual(result.status, ConversionStatus.EXHAUSTED_RETRY)

    def test_orchestrator_enables_tie_equivalence_for_mysql_to_pg(self):
        """Verify MySQLToPGQueryConverter triggers SQL-provenance Check 3 and tie-breaker retries."""
        mysql_converter = MySQLToPGQueryConverter()
        orchestrator = QueryConversionOrchestrator(
            converter=mysql_converter,
            source_connection={},
            dest_connection={},
            source_schema="CREATE TABLE t (grp VARCHAR(10), v INT);",
            dest_schema='CREATE TABLE "t" ("grp" VARCHAR(10), "v" INT);',
            max_attempts=1,
        )
        source_sql = (
            "SELECT DISTINCT grp, MIN(v) OVER (ORDER BY grp "
            "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS w FROM t"
        )
        candidate_pg_sql = (
            'SELECT DISTINCT "grp", MIN("v") OVER (ORDER BY "grp" '
            'ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS "w" FROM "t"'
        )
        src_res = QueryResult(
            columns=["grp", "w"], rows=[("A", 1), ("A", 2)], total_rows=2
        )
        dest_res = QueryResult(
            columns=["grp", "w"], rows=[("A", 1)], total_rows=1
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
                orchestrator,
                "_check_shared_window_tiebreaker",
                return_value=(True, "All checks passed (shared window tie-breaker)"),
            ) as mock_tb,
            mock.patch(
                "db_convertor.query_conversion_orchestrator.gemini_inference",
                return_value=llm_reply,
            ),
        ):
            result = orchestrator.convert_query(
                source_sql, question_id="q2", verbose=False
            )
            mock_tb.assert_called_once_with(source_sql, candidate_pg_sql)
            self.assertEqual(result.status, ConversionStatus.RESULT_MATCHED)


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
