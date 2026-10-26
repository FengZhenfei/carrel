"""Additional static formats shared by scanning, routing and profiles."""
from pathlib import Path
import re

EXTRA_PROFILES = {
    ".r": "r-symbols-v1", ".ipynb": "notebook-cells-v1",
    ".rst": "rst-sections-v1", ".qmd": "qmd-sections-v1", ".rmd": "rmd-sections-v1",
    ".tex": "tex-sections-v1", ".bib": "bib-entries-v1", ".jsonl": "jsonl-records-v1",
    ".tsv": "tsv-table-v1", ".nf": "workflow-text-v1", ".cwl": "workflow-text-v1",
    ".wdl": "workflow-text-v1", ".feature": "feature-scenarios-v1",
    ".dot": "dot-source-v1", ".xsd": "xsd-definitions-v1", ".log": "log-text-v1",
}
EXTRA_TEXT_SUFFIXES = set(EXTRA_PROFILES) - {".ipynb", ".tsv"}
WORKFLOW_FILENAMES = {"Snakefile", "snakefile", "nextflow.config"}
_SHEBANG = re.compile(r"^#![^\n]*\b(?:python[23]?(?:\.\d+)?|bash|sh|zsh|Rscript|nextflow|cwl-runner|snakemake)(?:\s|$)")


def extensionless_script(path: Path) -> bool:
    if path.suffix:
        return False
    try:
        with path.open("rb") as handle:
            head = handle.read(512).split(b"\n", 1)[0]
        return bool(_SHEBANG.match(head.decode("ascii")))
    except (OSError, UnicodeDecodeError):
        return False
