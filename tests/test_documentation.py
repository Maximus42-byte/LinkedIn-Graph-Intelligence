from pathlib import Path
import re


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


def _table_cells(line: str) -> list[str]:
    """Escaped pipes inside a cell must not count as column boundaries."""
    assert line.startswith('|') and line.endswith('|')
    return re.split(r"(?<!\\)\|", line[1:-1])


def _is_separator(line: str) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in _table_cells(line))


def test_generated_report_tables_are_separate_and_rectangular() -> None:
    lines = Path('output/report.md').read_text().splitlines()
    table_count = 0
    for start, line in enumerate(lines):
        if not line.startswith('|') or (start > 0 and lines[start - 1].startswith('|')):
            continue
        end = start
        while end < len(lines) and lines[end].startswith('|'):
            end += 1
        block = lines[start:end]
        table_count += 1
        assert start > 0 and not lines[start - 1].strip(), f'table at line {start + 1} needs a blank line before it'
        assert end < len(lines) and not lines[end].strip(), f'table at line {start + 1} needs a blank line after it'
        assert len(block) >= 2 and _is_separator(block[1]), f'table at line {start + 1} needs a separator row'
        width = len(_table_cells(block[0]))
        for row in block[1:]:
            assert len(_table_cells(row)) == width, f'wrong column count at line {start + 1}'
        assert not any(_is_separator(row) for row in block[2:]), f'tables are glued together at line {start + 1}'
    assert table_count > 0
