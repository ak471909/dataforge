"""
Metrics collection for monitoring application performance.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import time


class MetricsCollector:
    """Collect and track application metrics."""
    
    def __init__(self):
        self.request_count = 0
        self.request_durations = []
        self.error_count = 0
        self.endpoint_stats = defaultdict(lambda: {
            "count": 0,
            "total_duration": 0,
            "errors": 0
        })
        self.start_time = datetime.utcnow()
    
    def record_request(
        self,
        endpoint: str,
        duration: float,
        status_code: int
    ):
        """Record a request."""
        self.request_count += 1
        self.request_durations.append(duration)
        
        stats = self.endpoint_stats[endpoint]
        stats["count"] += 1
        stats["total_duration"] += duration
        
        if status_code >= 400:
            self.error_count += 1
            stats["errors"] += 1
    
    def get_metrics(self) -> Dict:
        """Get current metrics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Calculate averages
        avg_duration = (
            sum(self.request_durations) / len(self.request_durations)
            if self.request_durations else 0
        )
        
        # Endpoint statistics
        endpoint_metrics = {}
        for endpoint, stats in self.endpoint_stats.items():
            endpoint_metrics[endpoint] = {
                "requests": stats["count"],
                "avg_duration": (
                    stats["total_duration"] / stats["count"]
                    if stats["count"] > 0 else 0
                ),
                "errors": stats["errors"],
                "error_rate": (
                    stats["errors"] / stats["count"] * 100
                    if stats["count"] > 0 else 0
                )
            }
        
        return {
            "uptime_seconds": round(uptime, 2),
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate_percent": (
                self.error_count / self.request_count * 100
                if self.request_count > 0 else 0
            ),
            "avg_response_time_ms": round(avg_duration * 1000, 2),
            "requests_per_minute": round(
                self.request_count / (uptime / 60) if uptime > 0 else 0,
                2
            ),
            "endpoints": endpoint_metrics
        }
    
    def reset(self):
        """Reset all metrics."""
        self.__init__()


# Global metrics collector
metrics = MetricsCollector()