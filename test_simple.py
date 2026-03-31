from zhipuai import ZhipuAI

API_KEY = "a547932fbd50469589727a75b9972f1a.I2bdyK01FbCD2dT3"

client = ZhipuAI(api_key=API_KEY)

print("Testing API connection...")
print(f"API Key: {API_KEY[:10]}...")

try:
    # Try a very simple request
    response = client.chat.completions.create(
        model="glm-5",
        messages=[
            {"role": "user", "content": "Hello"}
        ],
        max_tokens=50
    )

    print("\n✅ SUCCESS!")
    print(f"Response: {response.choices[0].message.content}")

    if hasattr(response, 'usage'):
        print(f"Tokens used: {response.usage.total_tokens}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print(f"\nError type: {type(e).__name__}")
