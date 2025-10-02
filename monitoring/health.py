"""
Health check system for monitoring application status.
"""

from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import psutil
import os


class HealthCheck:
    """Comprehensive health check system."""
    
    @staticmethod
    def check_disk_space(path: str = ".", threshold_gb: float = 1.0) -> Dict:
        """Check available disk space."""
        try:
            stat = psutil.disk_usage(path)
            available_gb = stat.free / (1024 ** 3)
            
            return {
                "status": "healthy" if available_gb > threshold_gb else "warning",
                "available_gb": round(available_gb, 2),
                "total_gb": round(stat.total / (1024 ** 3), 2),
                "percent_used": stat.percent
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @staticmethod
    def check_memory(threshold_percent: float = 90.0) -> Dict:
        """Check memory usage."""
        try:
            mem = psutil.virtual_memory()
            
            return {
                "status": "healthy" if mem.percent < threshold_percent else "warning",
                "percent_used": mem.percent,
                "available_gb": round(mem.available / (1024 ** 3), 2),
                "total_gb": round(mem.total / (1024 ** 3), 2)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @staticmethod
    def check_cpu(threshold_percent: float = 90.0) -> Dict:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                "status": "healthy" if cpu_percent < threshold_percent else "warning",
                "percent_used": cpu_percent,
                "cpu_count": psutil.cpu_count()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @staticmethod
    def check_directories(directories: List[str]) -> Dict:
        """Check if required directories exist and are writable."""
        results = {}
        
        for directory in directories:
            path = Path(directory)
            exists = path.exists()
            writable = os.access(path, os.W_OK) if exists else False
            
            results[directory] = {
                "exists": exists,
                "writable": writable,
                "status": "healthy" if (exists and writable) else "error"
            }
        
        return results
    
    @staticmethod
    def check_environment_variables(required_vars: List[str]) -> Dict:
        """Check if required environment variables are set."""
        results = {}
        
        for var in required_vars:
            value = os.getenv(var)
            results[var] = {
                "set": value is not None,
                "status": "healthy" if value else "error"
            }
        
        return results
    
    @staticmethod
    def get_system_info() -> Dict:
        """Get general system information."""
        return {
            "python_version": os.sys.version,
            "platform": os.sys.platform,
            "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown",
            "uptime_seconds": round(psutil.boot_time()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def comprehensive_check() -> Dict:
        """Run all health checks."""
        return {
            "status": "healthy",  # Will be updated based on checks
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "disk": HealthCheck.check_disk_space(),
                "memory": HealthCheck.check_memory(),
                "cpu": HealthCheck.check_cpu(),
                "directories": HealthCheck.check_directories([
                    "output", "logs", "temp"
                ]),
                "environment": HealthCheck.check_environment_variables([
                    "TDB_OPENAI_API_KEY", "API_KEY"
                ])
            },
            "system": HealthCheck.get_system_info()
        }