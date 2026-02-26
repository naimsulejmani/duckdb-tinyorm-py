"""
Unit tests for DuckDB 1.x compatibility changes:
  - QueryBuilder.or_where / where_in use positional ? placeholders
  - BaseRepository.update / exists_by_id / find_by_id / remove_by_id / remove_all
    use positional ? placeholders
  - MigrationManager.apply_migration / revert_migration use positional ? placeholders

All tests run against an in-memory DuckDB database.
"""
import asyncio
import unittest

import duckdb

from duckdb_tinyorm_py.decorators import entity, field, id_field, repository
from duckdb_tinyorm_py.migration import Migration, MigrationManager
from duckdb_tinyorm_py.repository import (
    BaseRepository,
    DuckDbConfig,
    DuckDbLocation,
    DuckDbRepository,
    QueryBuilder,
)


# ---------------------------------------------------------------------------
# Shared test entity / repository definitions
# ---------------------------------------------------------------------------

@entity(table_name="compat_items")
class CompatItem:
    """Simple entity used across all repository tests."""

    def __init__(self, item_id=None, name="", category="", score=0):
        self._item_id = item_id
        self._name = name
        self._category = category
        self._score = score

    @property
    @id_field("INTEGER")
    def item_id(self) -> int:
        return self._item_id

    @item_id.setter
    def item_id(self, value: int):
        self._item_id = value

    @property
    @field("VARCHAR")
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    @field("VARCHAR")
    def category(self) -> str:
        return self._category

    @category.setter
    def category(self, value: str):
        self._category = value

    @property
    @field("INTEGER")
    def score(self) -> int:
        return self._score

    @score.setter
    def score(self, value: int):
        self._score = value


@repository(CompatItem)
class CompatItemRepository(BaseRepository):
    pass


def _make_db(name: str) -> DuckDbRepository:
    """Create a fresh in-memory DuckDB instance with a unique name."""
    return DuckDbRepository(DuckDbConfig(name=name, location=DuckDbLocation.MEMORY))


# ---------------------------------------------------------------------------
# QueryBuilder tests  (no DB needed – only SQL generation)
# ---------------------------------------------------------------------------

class TestQueryBuilderPositionalParams(unittest.TestCase):
    """Verify QueryBuilder emits positional ? placeholders (not :param_name)."""

    def test_where_uses_positional_placeholder(self):
        sql, params = QueryBuilder("t").where("x", "=", 42).build()
        self.assertIn("x = ?", sql)
        self.assertNotIn(":param", sql)
        self.assertEqual(params, [42])

    def test_or_where_uses_positional_placeholder(self):
        sql, params = (
            QueryBuilder("t")
            .where("category", "=", "fruit")
            .or_where("category", "=", "veg")
            .build()
        )
        self.assertIn("?", sql)
        self.assertNotIn(":param", sql)
        self.assertEqual(params, ["fruit", "veg"])

    def test_or_where_without_prior_condition_acts_as_where(self):
        """or_where with no existing clauses should behave like where."""
        sql, params = QueryBuilder("t").or_where("name", "=", "Alice").build()
        self.assertIn("name = ?", sql)
        self.assertEqual(params, ["Alice"])

    def test_where_in_uses_positional_placeholders(self):
        sql, params = QueryBuilder("t").where_in("item_id", [1, 2, 3]).build()
        self.assertIn("item_id IN (?, ?, ?)", sql)
        self.assertNotIn(":param", sql)
        self.assertEqual(params, [1, 2, 3])

    def test_where_in_empty_list_produces_always_false(self):
        sql, params = QueryBuilder("t").where_in("item_id", []).build()
        self.assertIn("1 = 0", sql)
        self.assertEqual(params, [])

    def test_multiple_where_clauses_ordered_params(self):
        sql, params = (
            QueryBuilder("t")
            .where("a", "=", 1)
            .where("b", ">", 2)
            .build()
        )
        self.assertEqual(params, [1, 2])
        self.assertIn("a = ?", sql)
        self.assertIn("b > ?", sql)

    def test_where_in_with_one_value(self):
        sql, params = QueryBuilder("t").where_in("x", [99]).build()
        self.assertIn("x IN (?)", sql)
        self.assertEqual(params, [99])

    def test_or_where_sql_executes_against_in_memory_db(self):
        """Generated SQL with ? placeholders must be accepted by DuckDB 1.x.

        A raw duckdb connection is used here intentionally: QueryBuilder tests
        only need to verify the SQL syntax is accepted by DuckDB — no ORM layer
        is required.
        """
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t (category VARCHAR, name VARCHAR)")
        con.execute("INSERT INTO t VALUES ('fruit', 'Apple'), ('veg', 'Carrot')")
        sql, params = (
            QueryBuilder("t")
            .where("category", "=", "fruit")
            .or_where("name", "=", "Carrot")
            .build()
        )
        rows = con.execute(sql, params).fetchall()
        self.assertEqual(len(rows), 2)

    def test_where_in_sql_executes_against_in_memory_db(self):
        """Generated WHERE IN SQL must be accepted by DuckDB 1.x.

        Uses a raw duckdb connection for the same reason as the or_where test
        above — only SQL syntax/execution is under test here.
        """
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t (id INTEGER, name VARCHAR)")
        con.execute("INSERT INTO t VALUES (1,'A'),(2,'B'),(3,'C')")
        sql, params = QueryBuilder("t").where_in("id", [1, 3]).build()
        rows = con.execute(sql, params).fetchall()
        self.assertEqual(len(rows), 2)


# ---------------------------------------------------------------------------
# BaseRepository tests (async, in-memory DuckDB)
# ---------------------------------------------------------------------------

class TestBaseRepositoryCompatibility(unittest.IsolatedAsyncioTestCase):
    """Integration tests for BaseRepository with an in-memory DuckDB database."""

    async def asyncSetUp(self):
        self.db = _make_db(f"repo_compat_{id(self)}")
        self.repo = CompatItemRepository(self.db)
        await self.repo.init(drop_if_exists=True)

        # Seed three rows
        await self.repo.insert(CompatItem(item_id=1, name="Alpha", category="A", score=10))
        await self.repo.insert(CompatItem(item_id=2, name="Beta",  category="B", score=20))
        await self.repo.insert(CompatItem(item_id=3, name="Gamma", category="A", score=30))

    async def asyncTearDown(self):
        self.db.close()

    # --- exists_by_id ---

    async def test_exists_by_id_returns_true_for_existing(self):
        self.assertTrue(await self.repo.exists_by_id(1))

    async def test_exists_by_id_returns_false_for_missing(self):
        self.assertFalse(await self.repo.exists_by_id(999))

    # --- find_by_id ---

    async def test_find_by_id_returns_correct_entity(self):
        item = await self.repo.find_by_id(2)
        self.assertIsNotNone(item)
        self.assertEqual(item.name, "Beta")
        self.assertEqual(item.category, "B")

    async def test_find_by_id_returns_none_for_missing(self):
        item = await self.repo.find_by_id(999)
        self.assertIsNone(item)

    # --- update ---

    async def test_update_persists_changes(self):
        item = await self.repo.find_by_id(1)
        item.name = "AlphaUpdated"
        item.score = 99
        await self.repo.update(item)

        updated = await self.repo.find_by_id(1)
        self.assertEqual(updated.name, "AlphaUpdated")
        self.assertEqual(updated.score, 99)

    async def test_update_does_not_affect_other_rows(self):
        item = await self.repo.find_by_id(1)
        item.name = "AlphaUpdated"
        await self.repo.update(item)

        other = await self.repo.find_by_id(2)
        self.assertEqual(other.name, "Beta")

    # --- remove_by_id ---

    async def test_remove_by_id_deletes_the_row(self):
        await self.repo.remove_by_id(1)
        self.assertFalse(await self.repo.exists_by_id(1))

    async def test_remove_by_id_does_not_delete_other_rows(self):
        await self.repo.remove_by_id(1)
        self.assertTrue(await self.repo.exists_by_id(2))
        self.assertTrue(await self.repo.exists_by_id(3))

    # --- remove_all with criteria ---

    async def test_remove_all_with_criteria_deletes_matching_rows(self):
        await self.repo.remove_all({"category": "A"})
        self.assertFalse(await self.repo.exists_by_id(1))
        self.assertFalse(await self.repo.exists_by_id(3))

    async def test_remove_all_with_criteria_preserves_non_matching_rows(self):
        await self.repo.remove_all({"category": "A"})
        self.assertTrue(await self.repo.exists_by_id(2))

    async def test_remove_all_without_criteria_deletes_everything(self):
        await self.repo.remove_all()
        total = await self.repo.count()
        self.assertEqual(total, 0)

    # --- execute_query with QueryBuilder (or_where / where_in) ---

    async def test_execute_query_with_or_where(self):
        qb = (
            self.repo.query()
            .where("category", "=", "B")
            .or_where("name", "=", "Gamma")
        )
        results = await self.repo.execute_query(qb)
        names = {r.name for r in results}
        self.assertIn("Beta", names)
        self.assertIn("Gamma", names)
        self.assertNotIn("Alpha", names)

    async def test_execute_query_with_where_in(self):
        qb = self.repo.query().where_in("item_id", [1, 3])
        results = await self.repo.execute_query(qb)
        ids = {r.item_id for r in results}
        self.assertEqual(ids, {1, 3})


# ---------------------------------------------------------------------------
# MigrationManager tests (async, in-memory DuckDB)
# ---------------------------------------------------------------------------

class _CreateTableMigration(Migration):
    """A concrete migration used in tests."""

    def __init__(self):
        super().__init__("create_test_table", "1.0.0")

    async def up(self, db: DuckDbRepository):
        db.execute("CREATE TABLE IF NOT EXISTS migration_test_tbl (id INTEGER)")

    async def down(self, db: DuckDbRepository):
        db.execute("DROP TABLE IF EXISTS migration_test_tbl")


class TestMigrationManagerCompatibility(unittest.IsolatedAsyncioTestCase):
    """Integration tests for MigrationManager with an in-memory DuckDB database."""

    async def asyncSetUp(self):
        self.db = _make_db(f"mig_compat_{id(self)}")
        self.manager = MigrationManager(self.db)
        await self.manager.init()

    async def asyncTearDown(self):
        self.db.close()

    async def test_get_applied_migrations_initially_empty(self):
        migrations = await self.manager.get_applied_migrations()
        self.assertEqual(migrations, [])

    async def test_apply_migration_returns_true_on_first_apply(self):
        m = _CreateTableMigration()
        result = await self.manager.apply_migration(m)
        self.assertTrue(result)

    async def test_apply_migration_records_migration_in_table(self):
        m = _CreateTableMigration()
        await self.manager.apply_migration(m)
        applied = await self.manager.get_applied_migrations()
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]["name"], "create_test_table")
        self.assertEqual(applied[0]["version"], "1.0.0")

    async def test_apply_migration_returns_false_when_already_applied(self):
        m = _CreateTableMigration()
        await self.manager.apply_migration(m)
        result = await self.manager.apply_migration(m)
        self.assertFalse(result)

    async def test_revert_migration_returns_true_when_applied(self):
        m = _CreateTableMigration()
        await self.manager.apply_migration(m)
        result = await self.manager.revert_migration(m)
        self.assertTrue(result)

    async def test_revert_migration_removes_record_from_table(self):
        m = _CreateTableMigration()
        await self.manager.apply_migration(m)
        await self.manager.revert_migration(m)
        applied = await self.manager.get_applied_migrations()
        self.assertEqual(applied, [])

    async def test_revert_migration_returns_false_when_not_applied(self):
        m = _CreateTableMigration()
        result = await self.manager.revert_migration(m)
        self.assertFalse(result)

    async def test_apply_migrations_batch(self):
        class _Mig2(Migration):
            def __init__(self):
                super().__init__("second_migration", "2.0.0")
            async def up(self, db):
                db.execute("CREATE TABLE IF NOT EXISTS mig2_tbl (id INTEGER)")
            async def down(self, db):
                db.execute("DROP TABLE IF EXISTS mig2_tbl")

        count = await self.manager.apply_migrations([_CreateTableMigration(), _Mig2()])
        self.assertEqual(count, 2)
        applied = await self.manager.get_applied_migrations()
        self.assertEqual(len(applied), 2)

    async def test_revert_migrations_batch(self):
        class _Mig2(Migration):
            def __init__(self):
                super().__init__("second_migration", "2.0.0")
            async def up(self, db):
                db.execute("CREATE TABLE IF NOT EXISTS mig2_tbl (id INTEGER)")
            async def down(self, db):
                db.execute("DROP TABLE IF EXISTS mig2_tbl")

        migrations = [_CreateTableMigration(), _Mig2()]
        await self.manager.apply_migrations(migrations)
        count = await self.manager.revert_migrations(migrations)
        self.assertEqual(count, 2)
        applied = await self.manager.get_applied_migrations()
        self.assertEqual(applied, [])


if __name__ == "__main__":
    unittest.main()
