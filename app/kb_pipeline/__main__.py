from .cli import main

if __name__ == "__main__":
    # The exit code must reach the shell: kb-pipeline-scan.sh keys the
    # mirror-flag handling off 75, and the mass-vanish refusal returns 2.
    # A bare main() call swallowed every return value into exit 0.
    raise SystemExit(main())
