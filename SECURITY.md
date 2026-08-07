# Security policy

## Supported version

Security fixes are applied to the current `main` branch. Historical experiment
artifacts and frozen protocols remain unchanged for provenance.

## Reporting a vulnerability

Do not publish credentials, private model endpoints, or exploit details in a
public issue. Use GitHub's private vulnerability-reporting option when it is
available. Otherwise, open a minimal issue requesting a private maintainer
contact without including sensitive details.

Include the affected commit, operating system, reproduction boundary, and
whether the issue involves local files, artifact parsing, or network-facing
OpenPI transport. Reports are assessed by the project maintainer without a
guaranteed response SLA.

ArmBench is research software. Its guards and validators are not a safety
certification and must not be used as the sole protective layer for physical
robot hardware.
