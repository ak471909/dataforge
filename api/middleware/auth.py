"""
API Key authentication middleware.
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import os
from dotenv import load_dotenv

load_dotenv()


class AuthMiddleware(BaseHTTPMiddleware):
    """API key authentication middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.api_key = os.getenv("API_KEY", "dev-key-12345")
        print(f"[AUTH] Loaded API key from env: {self.api_key}")
        # Paths that don't require authentication
        # Paths that don't require authentication
        self.public_paths = {
            "/", 
            "/health", 
            "/docs", 
            "/openapi.json", 
            "/redoc",
            "/api/v1/monitoring/health/basic",
            "/api/v1/monitoring/health/detailed",
            "/api/v1/monitoring/metrics",
            "/api/v1/monitoring/status"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Check API key for protected endpoints."""
        # Skip auth for public paths
        if request.url.path in self.public_paths:
            return await call_next(request)
        
        # Check for API key in header
        api_key = request.headers.get("X-API-Key")

        # Debug logging
        print(f"[AUTH] Request path: {request.url.path}")
        print(f"[AUTH] Received API key: {api_key}")
        print(f"[AUTH] Expected API key: {self.api_key}")
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API key missing. Include 'X-API-Key' header."
            )
        
        if api_key != self.api_key:
            raise HTTPException(
                status_code=403,
                detail="Invalid API key"
            )
        
        return await call_next(request)