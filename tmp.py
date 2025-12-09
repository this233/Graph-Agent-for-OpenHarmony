import requests
import json

if __name__ == '__main__':
    url = "https://api.modelarts-maas.com/v1/chat/completions"  # API地址
    api_key = "BQm_Gkd1EoTcHkJfVf31dTWfMIOsW3_mKIDfM5j-MvvwNM5jNl9XnLOjvNjEOuDiIWoKb-DIphdRWt2gOoNwBw"  # 把MAAS_API_KEY替换成已获取的API Key

    # Send request.
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        "model": "qwen3-235b-a22b",  # model参数
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好"}
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)

    # Print result.
    print(response.status_code)
    print(response.text)