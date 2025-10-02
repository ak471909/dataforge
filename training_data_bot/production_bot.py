"""
Production Bot Configuration Loader
Loads production configuration from YAML file and creates production-ready bot instance.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from training_data_bot import TrainingDataBot

def load_production_config(config_path: str = "config/production.yaml") -> Dict[str, Any]:
    """
    Load production configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def create_production_bot(config_path: str = "config/production.yaml") -> TrainingDataBot:
    """
    Create production-ready bot with configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured TrainingDataBot instance
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_production_config(config_path)
    
    # Extract bot configuration
    bot_config = {
        "chunk_size": config['processing']['chunk_size'],
        "chunk_overlap": config['processing']['chunk_overlap'],
        "quality_threshold": config['quality']['threshold'],
        "max_workers": config['processing']['max_workers'],
    }
    
    # Create bot instance
    bot = TrainingDataBot(config=bot_config)
    
    # Get AI provider settings
    ai_provider = config['ai_provider']['default']
    
    # Get API key from environment
    if ai_provider == "openai":
        api_key = os.getenv('TDB_OPENAI_API_KEY')
        if not api_key:
            raise ValueError("TDB_OPENAI_API_KEY environment variable not set!")
        
        model = config['ai_provider']['openai']['model']
        temperature = config['ai_provider']['openai']['temperature']
        max_tokens = config['ai_provider']['openai']['max_tokens']
        
    elif ai_provider == "anthropic":
        api_key = os.getenv('TDB_ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("TDB_ANTHROPIC_API_KEY environment variable not set!")
        
        model = config['ai_provider']['anthropic']['model']
        temperature = config['ai_provider']['anthropic']['temperature']
        max_tokens = config['ai_provider']['anthropic']['max_tokens']
    else:
        raise ValueError(f"Unknown AI provider: {ai_provider}")
    
    # Set AI client
    bot.set_ai_client(
        provider=ai_provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return bot

def validate_production_config(config_path: str = "config/production.yaml") -> bool:
    """
    Validate production configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    config = load_production_config(config_path)
    
    # Required sections
    required_sections = [
        'application', 'ai_provider', 'processing', 
        'quality', 'logging', 'storage'
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate processing config
    processing = config['processing']
    if processing['chunk_overlap'] >= processing['chunk_size']:
        raise ValueError("chunk_overlap must be less than chunk_size")
    
    # Validate quality config
    quality = config['quality']
    if not 0.0 <= quality['threshold'] <= 1.0:
        raise ValueError("quality threshold must be between 0.0 and 1.0")
    
    # Validate AI provider
    default_provider = config['ai_provider']['default']
    if default_provider not in config['ai_provider']:
        raise ValueError(f"Default provider '{default_provider}' not configured")
    
    return True

def print_production_config(config_path: str = "config/production.yaml"):
    """
    Print production configuration in a readable format.
    
    Args:
        config_path: Path to configuration file
    """
    config = load_production_config(config_path)
    
    print("\n" + "="*60)
    print("PRODUCTION CONFIGURATION")
    print("="*60)
    
    print(f"\nApplication:")
    print(f"  Name: {config['application']['name']}")
    print(f"  Version: {config['application']['version']}")
    print(f"  Environment: {config['application']['environment']}")
    
    print(f"\nAI Provider:")
    print(f"  Default: {config['ai_provider']['default']}")
    print(f"  Model: {config['ai_provider'][config['ai_provider']['default']]['model']}")
    
    print(f"\nProcessing:")
    print(f"  Chunk Size: {config['processing']['chunk_size']}")
    print(f"  Chunk Overlap: {config['processing']['chunk_overlap']}")
    print(f"  Max Workers: {config['processing']['max_workers']}")
    print(f"  Max Concurrent: {config['processing']['max_concurrent']}")
    
    print(f"\nQuality:")
    print(f"  Threshold: {config['quality']['threshold']}")
    print(f"  Filtering: {config['quality']['enable_filtering']}")
    
    print(f"\nLogging:")
    print(f"  Level: {config['logging']['level']}")
    print(f"  File: {config['logging']['file']}")
    
    print(f"\nStorage:")
    print(f"  Output Dir: {config['storage']['output_dir']}")
    print(f"  Default Format: {config['storage']['default_format']}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    """Test the production configuration."""
    import asyncio
    
    async def test_production_bot():
        print("Testing Production Bot Configuration...")
        
        # Validate config
        try:
            validate_production_config()
            print("✓ Configuration is valid")
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            return
        
        # Print config
        print_production_config()
        
        # Create bot
        try:
            bot = create_production_bot()
            print("✓ Production bot created successfully")
            
            # Get bot info
            info = bot.get_provider_info() if hasattr(bot, 'get_provider_info') else {}
            if info:
                print(f"✓ AI Provider: {info.get('provider', 'unknown')}")
                print(f"✓ Model: {info.get('model', 'unknown')}")
            
            # Cleanup
            await bot.cleanup()
            print("✓ Bot cleanup completed")
            
        except ValueError as e:
            print(f"❌ Bot creation failed: {e}")
            print("\nMake sure to set the required environment variables:")
            print("  - TDB_OPENAI_API_KEY (for OpenAI)")
            print("  - TDB_ANTHROPIC_API_KEY (for Anthropic)")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    asyncio.run(test_production_bot())  