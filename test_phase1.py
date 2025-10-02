"""Quick Phase 1 validation script."""
import asyncio
from pathlib import Path

async def main():
    print("Phase 1 Deployment Tests\n" + "="*50)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Files exist
    print("\n1. Checking required files...")
    required_files = [
        "setup.py",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "config/production.yaml",
        "scripts/production_bot.py",
        "README.md",
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
            tests_passed += 1
        else:
            print(f"  ✗ {file} MISSING")
            tests_failed += 1
    
    # Test 2: Documents directory
    print("\n2. Checking documents directory...")
    if Path("documents").exists():
        doc_count = len(list(Path("documents").glob("*.txt")))
        print(f"  ✓ documents/ exists ({doc_count} files)")
        tests_passed += 1
    else:
        print("  ✗ documents/ missing")
        tests_failed += 1
    
    # Test 3: .env file
    print("\n3. Checking .env file...")
    if Path(".env").exists():
        print("  ✓ .env exists")
        tests_passed += 1
    else:
        print("  ✗ .env missing")
        tests_failed += 1
    
    # Test 4: Output files
    print("\n4. Checking output files...")
    output_files = list(Path("output").glob("*.jsonl"))
    if len(output_files) > 0:
        print(f"  ✓ {len(output_files)} output files created")
        tests_passed += 1
    else:
        print("  ✗ No output files found")
        tests_failed += 1
    
    # Test 5: Package installation
    print("\n5. Checking package installation...")
    try:
        import training_data_bot
        print("  ✓ Package importable")
        tests_passed += 1
    except ImportError:
        print("  ✗ Package not installed")
        tests_failed += 1
    
    # Summary
    print("\n" + "="*50)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Success Rate: {tests_passed}/{tests_passed + tests_failed}")
    
    if tests_failed == 0:
        print("\n✓ PHASE 1 COMPLETE - Ready for Phase 2!")
    else:
        print(f"\n✗ Fix {tests_failed} issue(s) before Phase 2")

if __name__ == "__main__":
    asyncio.run(main())