import os
import json
from openai import OpenAI
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from openai import AzureOpenAI  # My addition
client = AzureOpenAI(
    api_version="2025-01-01-preview",
    azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
    # azure_deployment="gpt-4o-mini-deployment",
    azure_ad_token=os.environ.get("AZURE_OPENAI_API_KEY")
)

# Function Implementations
def calculate(expression: str):
    """
    Performs calculation on the given expression.
    """
    try:
        result = eval(expression)  # Use eval with caution!
        return {"expression": expression, "result": str(result)}
    except (SyntaxError, NameError, TypeError) as e:
        return {"expression": expression, "error": f"Calculation error: {e}"}

# Define custom tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Performs mathematical calculation on a given expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '2 + 2 * 3'",
                    }
                },
                "required": ["expression"],
            },
        }
    },
]

available_functions = {
    "calculate": calculate,
}

# Function to process messages and handle function calls
def get_completion_from_messages(messages, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,  # Custom tools
        tool_choice="auto"  # Allow AI to decide if a tool should be called
    )

    response_message = response.choices[0].message

    print("First response:", response_message)

    if response_message.tool_calls:
        # Find the tool call content
        tool_call = response_message.tool_calls[0]

        # Extract tool name and arguments
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        tool_id = tool_call.id
        
        # Call the function
        function_to_call = available_functions[function_name]
        function_response = function_to_call(**function_args)

        print(function_response)

        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(function_args),
                    }
                }
            ]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "name": function_name,
            "content": json.dumps(function_response),
        })

        # Second call to get final response based on function output
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        final_answer = second_response.choices[0].message

        print("Second response:", final_answer)
        return final_answer

    return "No relevant function call found."

# Example usage
if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that can perform mathematical calculations."},
        {"role": "user", "content": "What is 52 * 0 + 2 * 3?"},
    ]

    response = get_completion_from_messages(messages)
    print("--- Full response: ---")
    pprint(response)
    print("--- Response text: ---")
    print(response.content)
