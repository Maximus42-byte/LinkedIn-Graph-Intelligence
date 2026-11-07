from pathlib import Path


def _table_after(text: str, heading: str) -> str:
    tail = text.split(heading, 1)[1]
    start = tail.index('|')
    lines = []
    for line in tail[start:].splitlines():
        if not line.startswith('|'):
            break
        lines.append(line)
    return '\n'.join(lines)


def test_readme_excerpts_are_exact_generated_report_tables() -> None:
    report = Path('output/report.md').read_text()
    readme = Path('README.md').read_text()
    for report_heading, readme_heading in (
        ('## Key people and roles', '### Top key people'),
        ('## Method comparison', '### Centrality method agreement'),
        ('## Validation and stability', '### Stability summary'),
    ):
        assert _table_after(readme, readme_heading) == _table_after(report, report_heading)
    assert 'synthetic' in readme.lower()
    assert 'not statistically meaningful' in readme
