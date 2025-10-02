import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load .env from the same directory as this script
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

from training_data_bot.ai import AIClient


load_dotenv()


async def test_openai_connection():
    """Test OpenAI API connection and functionality."""
    print("\n" + "="*60)
    print("Testing OpenAI API Connection")
    print("="*60)
   
    api_key = os.getenv("TDB_OPENAI_API_KEY")
    if not api_key:
        print("❌ TDB_OPENAI_API_KEY not found in .env")
        return
   
    try:
        # Initialize client
        client = AIClient(
            provider="openai",
            api_key=api_key,
            model="gpt-3.5-turbo"  # Cheapest option
        )
        print(f"✓ Client initialized: {client}")
       
        # Test 1: Simple generation
        print("\nTest 1: Simple Generation")
        response = await client.generate(
            prompt="What is 2+2? Answer in one sentence.",
            max_tokens=50
        )
        print(f"✓ Response: {response.content}")
        print(f"✓ Tokens used: {response.tokens_used}")
        print(f"✓ Response time: {response.response_time:.2f}s")
       
        # Test 2: Cost calculation
        print("\nTest 2: Cost Estimation")
        cost = client.estimate_cost(
            response.metadata['prompt_tokens'],
            response.metadata['completion_tokens']
        )
        print(f"✓ Prompt tokens: {response.metadata['prompt_tokens']}")
        print(f"✓ Completion tokens: {response.metadata['completion_tokens']}")
        print(f"✓ Estimated cost: ${cost:.4f}")
       
        # Test 3: With system prompt
        print("\nTest 3: With System Prompt")
        response2 = await client.generate(
            prompt="Explain AI in 10 words.",
            system_prompt="You are a helpful assistant that gives concise answers.",
            max_tokens=30
        )
        print(f"✓ Response: {response2.content}")
       
        # Test 4: Token counting
        print("\nTest 4: Token Counting")
        test_text = "The Training Data Bot is an enterprise-grade system."
        tokens = client.count_tokens(test_text)
        print(f"✓ Text: '{test_text}'")
        print(f"✓ Estimated tokens: {tokens}")
       
        await client.close()
       
        print("\n" + "="*60)
        print("✓ ALL OPENAI TESTS PASSED!")
        print("="*60)
       
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_openai_connection())
