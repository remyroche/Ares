"""
GLM-5 using OpenAI-compatible client
Try using OpenAI-compatible endpoint as mentioned in Z.ai documentation
"""

from openai import OpenAI

API_KEY = "a547932fbd50469589727a75b9972f1a.I2bdyK01FbCD2dT3"

def test_glm5_openai_compat():
    """Test GLM-5 using OpenAI-compatible client"""
    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )

        print("🚀 Testing GLM-5 via OpenAI-compatible endpoint...")
        print("="*50)

        response = client.chat.completions.create(
            model="glm-5",
            messages=[{"role": "user", "content": "Say 'GLM-5 is working!' in one sentence"}]
        )

        print("✅ Success!")
        print("Response:", response.choices[0].message.content)
        print("="*50)

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_glm5_openai_compat()
