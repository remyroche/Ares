"""
GLM-5 Integration Script
Now that your API key is configured with coding-helper, you can use GLM-5 directly
"""

from zhipuai import ZhipuAI

# Use the API key that was configured with coding-helper
API_KEY = "a547932fbd50469589727a75b9972f1a.I2bdyK01FbCD2dT3"

def chat_with_glm5(prompt, temperature=0.7):
    """Chat with GLM-5"""
    client = ZhipuAI(api_key=API_KEY)

    try:
        response = client.chat.completions.create(
            model="glm-5",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Test GLM-5
    print("🚀 GLM-5 Test\n")
    print("="*50)

    result = chat_with_glm5("Explain what you can do in 2-3 sentences.")

    print(result)
    print("\n" + "="*50)
    print("✅ GLM-5 is ready to use!")
