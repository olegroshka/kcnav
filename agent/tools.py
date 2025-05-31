# agent/tools.py
import math
# import operator # Not strictly needed for current CalculatorTool
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


# For a more robust calculator, consider asteval or similar
# from asteval import Interpreter

class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def execute(self, input_str: str) -> str:
        pass


class CalculatorTool(Tool):
    @property
    def name(self) -> str:
        return "Calculator"

    @property
    def description(self) -> str:
        return ("A calculator for arithmetic operations (+, -, *, /, **). "
                "Input should be a mathematical expression string (e.g., '10 + 5 * (3 - 1)**2'). "
                "Use ** for exponentiation. Ensure proper spacing and parentheses for clarity.")

    def __init__(self):
        # self.interp = Interpreter() # For asteval
        pass

    def execute(self, input_str: str) -> str:
        safe_expr = input_str.replace('^', '**')  # Python uses **

        # More restrictive character set for basic safety with eval
        # Allows numbers, standard operators, parentheses, and decimal points.
        # Does NOT allow function calls like math.sqrt directly in the string for this basic eval.
        allowed_chars = "0123456789+-*/(). "
        # Check for any characters not in the allowed set
        if not all(c in allowed_chars for c in safe_expr.replace(" ", "")):  # Ignore spaces for this check
            # A more advanced parser would be better here.
            # This is a very basic attempt to prevent arbitrary code.
            # Search for any letters (potential function calls or variables)
            import re
            if re.search(r"[a-zA-Z]", safe_expr):
                return f"Error: Expression '{input_str}' contains disallowed characters or functions. Only numbers and basic operators (+-*/()**) are allowed."

        try:
            # WARNING: eval() is powerful and can be a security risk if input_str is not sanitized.
            # The character check above is a minimal attempt. A true sandboxed parser is better.
            # For example, using asteval:
            # self.interp.eval(safe_expr)
            # if self.interp.error:
            #     err_msg = "".join(self.interp.error_msg).strip()
            #     self.interp.error = [] # Clear errors
            #     return f"Error: {err_msg}"
            # result = self.interp.symtable['ans'] if 'ans' in self.interp.symtable else self.interp.symtable.get('_', None)

            # Using a very restricted eval:
            result = eval(safe_expr, {"__builtins__": {}}, {
                # No math functions exposed directly here for simplicity and safety with basic eval
                # If specific functions are needed, they must be explicitly added and validated.
            })

            if not isinstance(result, (int, float)):
                return f"Error: Evaluation resulted in non-numeric type ({type(result)})."

            # Format to avoid ".0" for integers
            if isinstance(result, float) and result.is_integer():
                return str(int(result))
            return str(result)
        except SyntaxError:
            return f"Error: Invalid syntax in expression: '{input_str}'"
        except ZeroDivisionError:
            return "Error: Division by zero."
        except OverflowError:
            return "Error: Calculation resulted in overflow."
        except NameError as e:  # Catch undefined variables/functions if any slip through
            return f"Error: Undefined name in expression: {e}"
        except Exception as e:
            return f"Error: Could not evaluate expression '{input_str}'. Details: {type(e).__name__} - {e}"


class ToolManager:
    def __init__(self, tools: Optional[List[Tool]] = None):
        if tools is None:
            tools = [CalculatorTool()]
        self.tools: Dict[str, Tool] = {tool.name.lower(): tool for tool in tools}
        print(f"ToolManager: Initialized with tools: {list(self.tools.keys())}")

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name.lower())

    def execute_tool(self, name: str, input_str: str) -> str:
        tool = self.get_tool(name)
        if tool:
            try:
                if not input_str.strip() and tool.name.lower() == "calculator":  # Calculator needs input
                    return f"Error: Input for tool '{name}' cannot be empty."
                return tool.execute(input_str)
            except Exception as e:
                print(f"ERROR: Unexpected exception during {name}.execute: {e}")
                return f"Error: Tool '{name}' encountered an issue. ({type(e).__name__})"
        else:
            available_tools = list(self.tools.keys())
            return f"Error: Tool '{name}' not found. Available tools: {available_tools}"

    def get_tool_descriptions(self) -> str:
        if not self.tools:
            return "No tools available."
        descriptions = ["Available Tools:"]
        for name in sorted(self.tools.keys()):  # Sort for consistent prompt order
            tool = self.tools[name]
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)
