import os
import json
import math
import numpy as np
from openai import OpenAI
from pprint import pprint
from dotenv import load_dotenv
from scipy.spatial import ConvexHull

# Load environment variables
load_dotenv()

# Initialize OpenAI client
# client = OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )

from openai import AzureOpenAI  # My addition
client = AzureOpenAI(
    api_version="2025-01-01-preview",
    azure_endpoint=os.environ.get("BUDWISE_ENDPOINT"),
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


def calculate_sphere_volume(radius: float):
    """
    Calculate the volume of a sphere.
    Formula: V = (4/3) * π * r³
    """
    try:
        volume = (4/3) * math.pi * (radius ** 3)
        return {
            "shape": "sphere",
            "radius": radius,
            "volume": round(volume, 6),
            "unit": "cubic units"
        }
    except Exception as e:
        return {"error": f"Error calculating sphere volume: {e}"}


def calculate_sphere_surface_area(radius: float):
    """
    Calculate the surface area of a sphere.
    Formula: A = 4 * π * r²
    """
    try:
        surface_area = 4 * math.pi * (radius ** 2)
        return {
            "shape": "sphere",
            "radius": radius,
            "surface_area": round(surface_area, 6),
            "unit": "square units"
        }
    except Exception as e:
        return {"error": f"Error calculating sphere surface area: {e}"}


def calculate_cube_volume(side_length: float):
    """
    Calculate the volume of a cube.
    Formula: V = s³
    """
    try:
        volume = side_length ** 3
        return {
            "shape": "cube",
            "side_length": side_length,
            "volume": round(volume, 6),
            "unit": "cubic units"
        }
    except Exception as e:
        return {"error": f"Error calculating cube volume: {e}"}


def calculate_cylinder_volume(radius: float, height: float):
    """
    Calculate the volume of a cylinder.
    Formula: V = π * r² * h
    """
    try:
        volume = math.pi * (radius ** 2) * height
        return {
            "shape": "cylinder",
            "radius": radius,
            "height": height,
            "volume": round(volume, 6),
            "unit": "cubic units"
        }
    except Exception as e:
        return {"error": f"Error calculating cylinder volume: {e}"}


def calculate_rectangle_area(length: float, width: float):
    """
    Calculate the area of a rectangle.
    Formula: A = length * width
    """
    try:
        area = length * width
        return {
            "shape": "rectangle",
            "length": length,
            "width": width,
            "area": round(area, 6),
            "unit": "square units"
        }
    except Exception as e:
        return {"error": f"Error calculating rectangle area: {e}"}


def calculate_circle_area(radius: float):
    """
    Calculate the area of a circle.
    Formula: A = π * r²
    """
    try:
        area = math.pi * (radius ** 2)
        return {
            "shape": "circle",
            "radius": radius,
            "area": round(area, 6),
            "unit": "square units"
        }
    except Exception as e:
        return {"error": f"Error calculating circle area: {e}"}


def calculate_convex_hull_volume(points: list):
    """
    Calculate the volume of a convex hull from a set of 3D points.
    Uses scipy's ConvexHull for complex geometric shapes.
    """
    try:
        # Convert to numpy array
        points_array = np.array(points)
        
        # Check if we have 3D points
        if points_array.shape[1] != 3:
            return {"error": "Points must be 3D (x, y, z coordinates)"}
        
        # Calculate convex hull
        hull = ConvexHull(points_array)
        
        return {
            "shape": "convex_hull",
            "num_points": len(points),
            "volume": round(hull.volume, 6),
            "surface_area": round(hull.area, 6),
            "unit": "cubic/square units"
        }
    except Exception as e:
        return {"error": f"Error calculating convex hull: {e}"}

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
    {
        "type": "function",
        "function": {
            "name": "calculate_sphere_volume",
            "description": "Calculate the volume of a sphere given its radius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "radius": {
                        "type": "number",
                        "description": "The radius of the sphere in any unit (mm, cm, m, etc.)",
                    }
                },
                "required": ["radius"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sphere_surface_area",
            "description": "Calculate the surface area of a sphere given its radius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "radius": {
                        "type": "number",
                        "description": "The radius of the sphere in any unit (mm, cm, m, etc.)",
                    }
                },
                "required": ["radius"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_cube_volume",
            "description": "Calculate the volume of a cube given its side length.",
            "parameters": {
                "type": "object",
                "properties": {
                    "side_length": {
                        "type": "number",
                        "description": "The side length of the cube in any unit (mm, cm, m, etc.)",
                    }
                },
                "required": ["side_length"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_cylinder_volume",
            "description": "Calculate the volume of a cylinder given its radius and height.",
            "parameters": {
                "type": "object",
                "properties": {
                    "radius": {
                        "type": "number",
                        "description": "The radius of the cylinder base",
                    },
                    "height": {
                        "type": "number",
                        "description": "The height of the cylinder",
                    }
                },
                "required": ["radius", "height"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_rectangle_area",
            "description": "Calculate the area of a rectangle given its length and width.",
            "parameters": {
                "type": "object",
                "properties": {
                    "length": {
                        "type": "number",
                        "description": "The length of the rectangle",
                    },
                    "width": {
                        "type": "number",
                        "description": "The width of the rectangle",
                    }
                },
                "required": ["length", "width"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_circle_area",
            "description": "Calculate the area of a circle given its radius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "radius": {
                        "type": "number",
                        "description": "The radius of the circle",
                    }
                },
                "required": ["radius"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_convex_hull_volume",
            "description": "Calculate the volume and surface area of a convex hull from 3D points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "points": {
                        "type": "array",
                        "description": "Array of 3D points [[x1,y1,z1], [x2,y2,z2], ...] defining the shape",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3
                        }
                    }
                },
                "required": ["points"],
            },
        }
    },
]

available_functions = {
    "calculate": calculate,
    "calculate_sphere_volume": calculate_sphere_volume,
    "calculate_sphere_surface_area": calculate_sphere_surface_area,
    "calculate_cube_volume": calculate_cube_volume,
    "calculate_cylinder_volume": calculate_cylinder_volume,
    "calculate_rectangle_area": calculate_rectangle_area,
    "calculate_circle_area": calculate_circle_area,
    "calculate_convex_hull_volume": calculate_convex_hull_volume,
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
        {"role": "system", "content": "You are a helpful AI assistant that can perform mathematical calculations and geometric computations for 2D and 3D shapes."},
        # {"role": "user", "content": "What is the volume of a sphere with radius of 1mm?"},
        {"role": "user", "content": "What is the surface of a 3D cube with side of 1.1mm?"},
    ]

    response = get_completion_from_messages(messages)
    print("--- Full response: ---")
    pprint(response)
    print("--- Response text: ---")
    print(response.content)
