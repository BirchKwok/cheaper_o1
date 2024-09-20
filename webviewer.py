from spinesUtils.asserts import ParameterTypeAssert, ParameterValuesAssert
import requests
import uuid


with open("api_key", "r") as f:
    api_key = f.read().strip()


@ParameterTypeAssert({
    "search_str": str,
    "return_type": str
})
@ParameterValuesAssert({
    "return_type": ("text", "json")
})
def web_viewer(search_str: str, return_type: str = "text"):
    """
    web 搜索

    Parameters:
        search_str: 搜索内容
        return_type: 返回类型，text 或 json

    Returns:
        (str | dict): 搜索结果
    """
    msg = [
        {
            "role": "user",
            "content": search_str
        }
    ]
    tool = "web-search-pro"
    url = "https://open.bigmodel.cn/api/paas/v4/tools"
    request_id = str(uuid.uuid4())
    data = {
        "request_id": request_id,
        "tool": tool,
        "stream": False,
        "messages": msg
    }

    resp = requests.post(
        url,
        json=data,
        headers={'Authorization': api_key},
        timeout=300
    )
    
    return resp.content.decode() if return_type == "text" else resp.json()