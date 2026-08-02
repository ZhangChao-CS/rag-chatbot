class ToolRegistry:

    def __init__(self):
        self.tools = {}

    def register(self, tool):

        if tool.name in self.tools:
            raise ValueError(f"Tool {tool.name} already exists")

        self.tools[tool.name] = tool

    def get(self, name):

        tool = self.tools.get(name)

        if tool is None:
            raise ValueError(f"Tool {name} not found")

        return tool

    def list_tools(self):

        return list(self.tools.keys())

    def get_tool_schema(self):

        result=[]

        for tool in self.tools.values():

            result.append({
                "name":
                    tool.name,

                "description":
                    tool.description,

                "parameters":
                    tool.args_schema.model_json_schema()
            })

        return result
