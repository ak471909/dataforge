"""
Diagnostic tool for troubleshooting the Training Data Bot.
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from monitoring.health import HealthCheck
from training_data_bot import TrainingDataBot
import os


async def run_diagnostics():
    """Run comprehensive diagnostics."""
    print("="*60)
    print("TRAINING DATA BOT - DIAGNOSTICS")
    print("="*60)
    
    # System Health
    print("\n1. SYSTEM HEALTH")
    print("-"*60)
    health = HealthCheck.comprehensive_check()
    
    print(f"Disk Space: {health['checks']['disk']['status']}")
    print(f"  Available: {health['checks']['disk']['available_gb']} GB")
    
    print(f"Memory: {health['checks']['memory']['status']}")
    print(f"  Used: {health['checks']['memory']['percent_used']}%")
    
    print(f"CPU: {health['checks']['cpu']['status']}")
    print(f"  Used: {health['checks']['cpu']['percent_used']}%")
    
    # Environment Variables
    print("\n2. ENVIRONMENT VARIABLES")
    print("-"*60)
    for var, info in health['checks']['environment'].items():
        status = "✓" if info['set'] else "✗"
        print(f"{status} {var}: {'Set' if info['set'] else 'Not Set'}")
    
    # Directories
    print("\n3. DIRECTORIES")
    print("-"*60)
    for dir_name, info in health['checks']['directories'].items():
        status = "✓" if info['status'] == 'healthy' else "✗"
        print(f"{status} {dir_name}: Exists={info['exists']}, Writable={info['writable']}")
    
    # Bot Initialization
    print("\n4. BOT INITIALIZATION")
    print("-"*60)
    try:
        bot = TrainingDataBot()
        print("✓ Bot initialized successfully")
        
        # Test AI client
        openai_key = os.getenv("TDB_OPENAI_API_KEY")
        if openai_key:
            bot.set_ai_client(provider="openai", api_key=openai_key)
            print("✓ AI client configured")
        else:
            print("✗ No OpenAI API key found")
        
        # Get statistics
        stats = bot.get_statistics()
        print(f"  Documents loaded: {stats['documents']['total']}")
        print(f"  Datasets created: {stats['datasets']['total']}")
        
        await bot.cleanup()
        
    except Exception as e:
        print(f"✗ Bot initialization failed: {e}")
    
    print("\n" + "="*60)
    print("DIAGNOSTICS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_diagnostics())