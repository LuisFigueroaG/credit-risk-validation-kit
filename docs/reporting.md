# Reporting

Reports are self-contained HTML files generated from aggregate metrics and
tables. They avoid raw customer rows and sensitive identifiers.

JSON output contains metrics, checks, metadata, config, and aggregate tables.

Do not use personal data, customer names, national identifiers, addresses,
phone numbers, email addresses, or sensitive identifiers in public reports.

When `report.anonymize` is enabled, CRVK creates a non-mutating export projection
for HTML, JSON, CSV/Parquet, and model cards. Configured column names and segment
values receive deterministic aliases, and free-text model ownership and purpose
fields are redacted. This is pseudonymization of generated artifacts, not a claim
that aggregate outputs are immune to re-identification. Set `anonymize: false`
only when original labels are required and the destination is trusted.

Drift HTML reports display current values as the primary KPI while retaining the
reference value and delta. Reference and current lift and calibration tables are
kept as separate artifacts.
