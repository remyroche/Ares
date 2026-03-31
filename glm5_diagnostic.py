"""
GLM-5 Diagnostic Script
Checks API key status and available models
"""

import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI

# Load environment variables
load_dotenv()

api_key = os.getenv("ZHIPUAI_API_KEY")
print(f"🔑 API Key (first 20 chars): {api_key[:20]}...")

client = ZhipuAI(api_key=api_key)

print("\n📊 Testing different models...")

# Test different model variations
models_to_test = [
    "glm-5",
    "glm-4",
    "glm-4-plus",
    "glm-4-flash",
    "glm-3-turbo"
]

for model in models_to_test:
    print(f"\n🧪 Testing model: {model}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10
        )
        print(f"✅ {model} works!")
        print(f"Response: {resp.choices[0].message.content[:50]}...")
        break  # Found a working model
    except Exception as e:
        error_msg = str(e)
        print(f"❌ {model} failed: {error_msg[:100]}")

print("\n" + "="*50)
print("💡 If all models fail with 'balance insufficient',")
print("   you need to add credits at: https://open.bigmodel.cn/")
