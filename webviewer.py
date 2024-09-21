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
    if not search_str:
        return ""
    
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
    
    try:
        resp = resp.json()
    except Exception as e:
        return resp.text

    try:
        resp = resp['choices'][0]['message']['tool_calls']
    except KeyError:
        return step_by_step_get_web_info(resp, "choices", "0", "message", "tool_calls")
    except Exception as e:
        return resp.text
    
    return str(resp) if return_type == "text" else resp


def step_by_step_get_web_info(resp: dict, *keys: str):
    """
    逐步获取网页信息
    """
    keys_link = []
    for key in keys:
        new_resp = resp.get(key, None)
        if new_resp is None:
            break
        keys_link.append(new_resp)
    
    if len(keys_link) == 0:
        return str(resp)
    
    # 逐步获取网页信息
    for key in keys_link:
        resp = resp[key]
    return str(resp)