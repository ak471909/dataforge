"""
Monitoring and observability endpoints.
"""

from fastapi import APIRouter
from typing import Dict
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from monitoring.health import HealthCheck
from monitoring.metrics import metrics
from api.dependencies import get_bot

router = APIRouter()


@router.get("/health/basic")
async def health_basic():
    """Basic health check."""
    try:
        bot = get_bot()
        return {
            "status": "healthy",
            "bot_initialized": True
        }
    except:
        return {
            "status": "unhealthy",
            "bot_initialized": False
        }


@router.get("/health/detailed")
async def health_detailed():
    """Detailed health check with system metrics."""
    return HealthCheck.comprehensive_check()


@router.get("/metrics")
async def get_metrics():
    """Get application metrics."""
    return metrics.get_metrics()


@router.get("/status")
async def get_status():
    """Get comprehensive application status."""
    bot = get_bot()
    bot_stats = bot.get_statistics()
    
    return {
        "status": "running",
        "health": HealthCheck.comprehensive_check(),
        "metrics": metrics.get_metrics(),
        "bot_statistics": bot_stats
    }


@router.post("/metrics/reset")
async def reset_metrics():
    """Reset metrics counters."""
    metrics.reset()
    return {"status": "success", "message": "Metrics reset"}