"""Scan local Git objects and candidate files without printing secret values."""
from pathlib import Path
import json
import re
import subprocess
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"),
    "aws_access_key": re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def main():
    secrets = {k: v.encode() for k, v in dotenv_values(ROOT / ".env").items() if v and len(v) > 12}
    findings = []
    def inspect(data, location):
        matches = ["configured:" + k for k, v in secrets.items() if v in data]
        matches += [name for name, pattern in PATTERNS.items() if pattern.search(data)]
        if matches: findings.append({"location": location, "types": matches})
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    for name in paths:
        path = ROOT / name
        if path.is_file(): inspect(path.read_bytes(), name)
    objects = subprocess.check_output(["git", "rev-list", "--objects", "--all"], cwd=ROOT, text=True).splitlines()
    process = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    scanned = 0
    for record in objects:
        oid, _, name = record.partition(" ")
        process.stdin.write((oid + "\n").encode()); process.stdin.flush()
        header = process.stdout.readline().split()
        if len(header) != 3: continue
        data = process.stdout.read(int(header[2])); process.stdout.read(1)
        if header[1] == b"blob":
            scanned += 1
            inspect(data, "history:" + name)
    process.stdin.close(); process.wait()
    summary = {"candidate_files": len(paths), "history_blobs": scanned,
               "configured_keys_compared": len(secrets), "findings": findings,
               "limits": "No remote fetch; limited patterns and currently configured keys only. Manual review required for matches."}
    print(json.dumps(summary, indent=2))
    return bool(findings)

if __name__ == "__main__":
    raise SystemExit(main())
