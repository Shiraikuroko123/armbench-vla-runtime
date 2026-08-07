"""Compatibility wrapper for the renamed temporal-alignment acceptance tool."""

from integrations.openpi.alignment_acceptance import (
    REPORT_SCHEMA_VERSION,
    build_alignment_acceptance,
    main,
)

__all__ = ["REPORT_SCHEMA_VERSION", "build_interview_acceptance", "main"]

build_interview_acceptance = build_alignment_acceptance


if __name__ == "__main__":
    raise SystemExit(main())
