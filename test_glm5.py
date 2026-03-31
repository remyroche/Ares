"""
GLM-5 Test Script
Tests ZhipuAI GLM-5 model connectivity and basic functionality
"""

import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI

# Load environment variables from .env file
load_dotenv()

def test_glm5():
    """Test basic GLM-5 connectivity and response"""

    # Get API key from environment
    api_key = os.getenv("ZHIPUAI_API_KEY")

    if not api_key or api_key == "your_zhipuai_api_key_here":
        print("❌ Error: ZHIPUAI_API_KEY not set in .env file")
        print("Please get your API key from https://open.bigmodel.cn/ and update the .env file")
        return False

    try:
        # Initialize the client
        print("🔑 Connecting to ZhipuAI...")
        client = ZhipuAI(api_key=api_key)

        # Test basic request
        print("📝 Testing GLM-5 model...")
        resp = client.chat.completions.create(
            model="glm-5",
            messages=[{"role": "user", "content": "Say 'GLM-5 is working!' in one sentence"}]
        )

        # Display result
        print("\n✅ Success! GLM-5 response:")
        print("=" * 50)
        print(resp.choices[0].message.content)
        print("=" * 50)

        # Show model info
        print(f"\n📊 Model used: {resp.model}")
        if hasattr(resp, 'usage') and resp.usage:
            print(f"📈 Tokens used: {resp.usage.total_tokens}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPossible issues:")
        print("1. Invalid API key - check your .env file")
        print("2. Network connectivity issue")
        print("3. API service temporarily unavailable")
        return False

if __name__ == "__main__":
    print("🚀 GLM-5 Test Script")
    print("=" * 50)
    success = test_glm5()
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! GLM-5 is ready to use.")
    else:
        print("❌ Tests failed. Please check the errors above.")
