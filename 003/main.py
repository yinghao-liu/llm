# my_server.py
from typing import Annotated, Literal
from fastmcp import FastMCP

mcp = FastMCP("My Tools")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""          # docstring 会自动变成工具描述
    return a + b



@mcp.tool
def get_weather(query: Annotated[str, "城市名，例如：北京"],
                ) -> str:
    """获取指定城市的天气"""
    return "晴, 25°C"

if __name__ == "__main__":
    mcp.run()                       # 默认 stdio