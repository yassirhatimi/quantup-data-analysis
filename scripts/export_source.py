"""Build a source-only review bundle; never publishes or changes Git history."""
from pathlib import Path
import argparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".review-build/quantlab-source.zip")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise SystemExit("Output must remain inside the project directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    names = {"LICENSE", "pyproject.toml", "requirements.txt", ".env.example", ".gitignore",
             "docs/DATA_POLICY.md", "docs/SOURCE_RELEASE.md", ".github/workflows/offline-checks.yml"}
    for folder in ("quantlab", "scripts", "tests", "examples", "dashboard", "gold_wave_analysis"):
        for path in (ROOT / folder).rglob("*.py"):
            if "__pycache__" not in path.parts:
                names.add(path.relative_to(ROOT).as_posix())
    readme = (ROOT / "docs/SOURCE_RELEASE.md").read_text(encoding="utf-8")
    readme = readme.replace("(../LICENSE)", "(LICENSE)").replace("(DATA_POLICY.md)", "(docs/DATA_POLICY.md)")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            archive.write(ROOT / name, name)
        archive.write(ROOT / "quantlab/predict/events.json", "quantlab/predict/events.json")
        archive.writestr("README.md", readme)
    print(f"Built source-only bundle: {output.name}; {len(names) + 2} files")
    print("Excluded: credentials, Git history, raw snapshots, notebook outputs and historical reports.")

if __name__ == "__main__":
    main()
