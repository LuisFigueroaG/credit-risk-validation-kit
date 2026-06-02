"""Standard status values used across validation checks."""

from enum import StrEnum


class Status(StrEnum):
    """Estado estandar para hallazgos y metricas."""

    OK = "OK"
    PASS = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    ERROR = "ERROR"


def worst_status(statuses: list[Status]) -> Status:
    """Devuelve el estado mas severo de una lista."""

    if not statuses:
        return Status.NOT_APPLICABLE
    order = {
        Status.ERROR: 5,
        Status.CRITICAL: 4,
        Status.WARNING: 3,
        Status.INSUFFICIENT_DATA: 2,
        Status.NOT_APPLICABLE: 1,
        Status.OK: 0,
    }
    return max(statuses, key=lambda status: order[status])
