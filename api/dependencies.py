"""
Shared dependencies for API routes.
"""

from fastapi import HTTPException
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training_data_bot import TrainingDataBot

# Global bot instance
_bot_instance = None


def set_bot(bot: "TrainingDataBot") -> None:
    """Set the global bot instance."""
    global _bot_instance
    _bot_instance = bot


def get_bot() -> "TrainingDataBot":
    """Get the global bot instance."""
    if _bot_instance is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")
    return _bot_instance