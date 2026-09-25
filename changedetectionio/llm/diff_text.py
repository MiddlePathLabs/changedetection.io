"""
The unified diff text sent to the LLM.

One builder for every caller (the worker's intent/summary pass and the diff page's Summary
button) so the model always sees the same shape of diff for the same pair of snapshots.
"""

import difflib


def build_llm_diff(a_text: str, b_text: str, ignore_whitespace: bool = False, context: int = 3) -> str:
    """Unified diff of two snapshots, one diff line per output line, without the ---/+++ header.

    Lines are split without their line endings and re-joined with '\\n'. Mixing
    splitlines(keepends=True) with lineterm='' (as the worker used to) glues the header lines
    onto the first hunk and, when the old snapshot has no trailing newline, glues its last
    removed line onto the next added line - the model then sees one removed line and no
    addition.
    """
    def _prep(text):
        lines = (text or '').splitlines()
        if ignore_whitespace:
            return [' '.join(line.split()) for line in lines]
        return lines

    lines = list(difflib.unified_diff(_prep(a_text), _prep(b_text), lineterm='', n=context))
    # Drop the '--- ' / '+++ ' file header; it carries no information here.
    if len(lines) >= 2 and lines[0].startswith('---') and lines[1].startswith('+++'):
        lines = lines[2:]
    return '\n'.join(lines)
