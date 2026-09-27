"""Service health endpoint."""

from fastapi import APIRouter

from app.models.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def get_health() -> HealthResponse:
    """Report that the API process is running."""
    return HealthResponse(status="healthy")
