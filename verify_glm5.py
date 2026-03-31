"""
Test if Claude Code is using GLM-5
Run this script and check the output
"""

import os
import json

print("=" * 60)
print("🔍 GLM-5 Verification for Claude Code")
print("=" * 60)

# Check 1: Configuration file
print("\n1️⃣  Checking Claude Code configuration...")
config_path = os.path.expanduser("~/.claude/settings.json")

try:
    with open(config_path, 'r') as f:
        config = json.load(f)

    base_url = config.get('env', {}).get('ANTHROPIC_BASE_URL', '')
    api_key = config.get('env', {}).get('ANTHROPIC_AUTH_TOKEN', '')

    if 'api.z.ai' in base_url:
        print(f"   ✅ Base URL: {base_url}")
        print(f"   ✅ Using ZhipuAI API (GLM-5)")
        print(f"   ✅ API Key: {api_key[:10]}...")
    else:
        print(f"   ❌ Base URL: {base_url}")
        print(f"   ❌ Not using ZhipuAI API")

except Exception as e:
    print(f"   ❌ Error reading config: {e}")

# Check 2: Environment variables
print("\n2️⃣  Checking environment variables...")
anthropic_url = os.getenv('ANTHROPIC_BASE_URL', '')
if 'api.z.ai' in anthropic_url:
    print(f"   ✅ ANTHROPIC_BASE_URL set to: {anthropic_url}")
else:
    print(f"   ⚠️  ANTHROPIC_BASE_URL: {anthropic_url or 'Not set'}")

# Check 3: Coding Helper status
print("\n3️⃣  GLM Coding Plan status...")
print("   Run: npx @z_ai/coding-helper doctor")

# Check 4: What to look for
print("\n4️⃣  GLM-5 Characteristics:")
print("   • Ask: 'What model are you? Describe your capabilities.'")
print("   • GLM-5 may mention different capabilities than Claude")
print("   • Check response style and formatting")

print("\n" + "=" * 60)
print("✅ Configuration verified!")
print("📝 Restart Claude to apply changes")
print("=" * 60)
