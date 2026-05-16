#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///

import json
import sys

import requests


def main() -> None:
    """
    clean-garbage V1 skill.

    作用：
    - 调用后端 /clean；
    - 不把业务失败吞掉；
    - 输出后端提供的结构化结果，供 Agent 更新 state 和做失败分类。
    """
    url = "http://127.0.0.1:5000/clean"

    try:
        response = requests.post(url, timeout=10)

        try:
            data = response.json()
        except Exception:
            data = {
                "status": "error",
                "result_type": "error_clean_invalid_response",
                "message": response.text,
                "http_status": response.status_code,
            }

        # 服务级错误：HTTP 500/连接异常等。
        # 业务级错误：目标不居中、无目标、不可达等，后端应返回 HTTP 200 + status=error。
        if response.status_code >= 500:
            data.setdefault("status", "error")
            data.setdefault("result_type", "error_clean_service_unavailable")
            data.setdefault("message", f"clean service HTTP {response.status_code}")
            print(json.dumps(data, ensure_ascii=False))
            sys.exit(1)

        data.setdefault("http_status", response.status_code)
        print(json.dumps(data, ensure_ascii=False))

        # 注意：业务失败不在这里 sys.exit(1)，否则 OpenClaw 可能只看到工具失败，
        # 看不到结构化 result_type。Agent 应读取 status/result_type 决定下一步。
        return

    except requests.exceptions.ConnectionError as e:
        print(
            json.dumps(
                {
                    "status": "error",
                    "result_type": "error_clean_service_unavailable",
                    "message": f"无法连接清扫服务: {e}",
                },
                ensure_ascii=False,
            )
        )
        sys.exit(1)
    except requests.exceptions.Timeout as e:
        print(
            json.dumps(
                {
                    "status": "error",
                    "result_type": "error_clean_timeout",
                    "message": f"清扫服务超时: {e}",
                },
                ensure_ascii=False,
            )
        )
        sys.exit(1)
    except Exception as e:
        print(
            json.dumps(
                {
                    "status": "error",
                    "result_type": "error_clean_unknown",
                    "message": f"清理模块启动失败: {e}",
                },
                ensure_ascii=False,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
