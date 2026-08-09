-- Model SQL
-- Reference documentation: https://docs.rilldata.com/developers/build/connectors/data-source/duckdb
-- @connector: duckdb
-- @output.connector: duckdb

-- ponytail: one row per PR (unnest pr_numbers). Per-repo lines/cost can't be
-- attributed to individual PRs, so split evenly by /N — sums stay correct.
with exploded as (
  select s.started_at, sr.session_id, sr.repository, sr.is_primary,
         unnest(cast(sr.pr_numbers as BIGINT[])) as pr_number,
         json_array_length(sr.pr_numbers) as n,
         sr.files_touched, sr.lines_added, sr.lines_removed,
         sr.lines_share, sr.stated_cost, sr.inferred_cost
  from session_repositories sr
    join sessions s on sr.session_id = s.session_id
  where sr.pr_numbers <> '[]'
)
select started_at, session_id, repository, is_primary, pr_number,
       files_touched  * 1.0 / n as files_touched,
       lines_added    * 1.0 / n as lines_added,
       lines_removed  * 1.0 / n as lines_removed,
       lines_share          / n as lines_share,
       stated_cost          / n as stated_cost,
       inferred_cost        / n as inferred_cost
from exploded
