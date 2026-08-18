# Security policy

## Supported version

Security fixes currently target the latest `0.1.x` release.

## Reporting

Please report a suspected vulnerability privately through GitHub's security
advisory feature instead of opening a public issue.

## Trust boundary

The compiler accepts a restricted TikZ subset, but XeLaTeX and Manim are still
external executables. Do not process untrusted source with unrestricted shell
escape or in an environment that contains sensitive files. Run untrusted jobs
inside an operating-system sandbox or disposable container.
