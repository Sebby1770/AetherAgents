from typing import Any, Callable, Dict, List
import inspect
import json
from pydantic import BaseModel

class ToolResult(BaseModel):
    content: str
    summary: str

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: List[Dict] = []

    def register(self, func: Callable, name: str = None, description: str = None):
        name = name or func.__name__
        self.tools[name] = func
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or (func.__doc__ or ""),
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }
        self.schemas.append(schema)

    def get_schemas(self) -> List[Dict]:
        return self.schemas

    async def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        func = self.tools[name]
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)
            return ToolResult(content=str(result), summary=str(result)[:300])
        except Exception as e:
            return ToolResult(content=f"Error: {e}", summary="Execution failed")