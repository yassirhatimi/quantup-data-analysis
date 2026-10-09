# Local release verification

Checked on 9 October 2026 using Python 3.13.13 on Windows.

- 10 replay and dashboard regression tests passed.
- 12 gold-analysis unit tests passed.
- 42 offline prediction-engine checks passed.
- The synthetic gold example ran and returned continued, fully_retraced,
  and end_of_data for its three hand-built paths.
- A wheel was built without downloading dependencies and installed into a
  separate local target. All seven data clients, the prediction engine and
  replay imported from that target; packaged event configuration was present.
- Candidate source files were compared with four locally configured key values
  and selected credential patterns. No matches were found. This limited check
  does not certify the absence of every possible secret.

Existing local dependencies were used. A clean dependency download, live API
access and GitHub Actions execution have not been verified in this review.
The installed-wheel import check used the existing environment's dependencies.

This copy was prepared from a source-only export with independent Git history.
The original project was preserved. The checks above describe local validation;
GitHub Actions records the results of subsequent hosted checks.
