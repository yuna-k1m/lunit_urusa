# Runtime health database

`lunit_health.db` is a compact, knowledge-only export of the sibling
`lunit-health-db` project. It contains Lunit MCP source/document catalog records
and no HealthBench examples, rubrics, ideal completions, or model outputs.

The retrieval-stage chat model can search it with the local
`health_db_search` tool. Set `LUNIT_HEALTH_DB_PATH` to use another compatible
database. The model-facing adapter always filters searches to `knowledge`.
