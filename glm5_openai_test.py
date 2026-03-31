"""
Test GLM-5 using OpenAI-compatible API
"""

from openai import OpenAI

API_KEY = "a547932fbd50469589727a75b9972f1a.I2bdyK01FbCD2dT3"

# Try different base URLs
base_urls = [
    "https://open.bigmodel.cn/api/paas/v4/",
    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "https://api.bigmodel.cn/api/paas/v4/",
]

for base_url in base_urls:
    print(f"\n{'='*60}")
    print(f"Trying base URL: {base_url}")
    print('='*60)

    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url=base_url
        )

        response = client.chat.completions.create(
            model="glm-5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=50
        )

        print("✅ SUCCESS!")
        print(f"Response: {response.choices[0].message.content}")
        break

    except Exception as e:
        print(f"❌ Error: {str(e)[:200]}")

print("\n" + "="*60)
