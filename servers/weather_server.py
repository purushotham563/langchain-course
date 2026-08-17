from mcp.server.fastmcp import FastMCP
mcp=FastMCP("weather")
@mcp.tool()
async def get_weather(location:str)->str:
    """Get weather for location"""
    return "Hot as hell"
if __name__=="__main__":
    mcp.run(transport="sse")