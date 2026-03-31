"""
Debug GLM-5 API response
"""

from openai import OpenAI
import json

API_KEY = "a547932fbd50469589727a75b9972f1a.I2bdyK01FbCD2dT3"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.bigmodel.cn/api/paas/v4/"
)

print("Testing GLM-5 with https://api.bigmodel.cn/api/paas/v4/")
print("="*60)

try:
    response = client.chat.completions.create(
        model="glm-5",
        messages=[{"role": "user", "content": "Say hello in one sentence"}],
        max_tokens=50
    )

    print("Response type:", type(response))
    print("\nFull response:")
    print(response)

    # Try to access the content
    if hasattr(response, 'choices'):
        print("\n✅ Content:", response.choices[0].message.content)
    else:
        print("\n⚠️  Response doesn't have 'choices' attribute")
        print("Direct string value:", str(response))

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
