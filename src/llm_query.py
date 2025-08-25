import os
import json
from openai import AsyncOpenAI, OpenAI

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def analyze_file_content_async(file_content, file_type, file_name):
    prompt = f"""
    Analyze this {file_type} file and extract the main topics and concepts covered.
    File name: {file_name}
    
    Content: {file_content[:2000]}...
    
    Return a JSON object with a "topics" array: {{"topics": ["topic1", "topic2"]}}
    """
    
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        return result["topics"]
    except (json.JSONDecodeError, KeyError):
        return ["Error parsing topics"]

async def generate_learning_path(learning_data):
    # Format the data for the prompt
    content_summary = []
    for record in learning_data:
        content_summary.append(f"{record['f.name']} ({record['f.type']}): {', '.join(record['topics'])}")
    
    prompt = f"""
    Create a structured learning path from this educational content. Organize it from beginner to advanced topics.
    
    Available content:
    {chr(10).join(content_summary)}
    
    Return a JSON object with this structure:
    {{
        "learning_path": [
            {{"step": 1, "title": "Getting Started", "files": ["README.md"], "description": "Introduction to fastai"}},
            {{"step": 2, "title": "First Project", "files": ["lesson1-pets.ipynb"], "description": "Build your first classifier"}}
        ]
    }}
    """
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        response_format={"type": "json_object"}
    )
    try:
        result = json.loads(response.choices[0].message.content)
        return result["learning_path"]
    except (json.JSONDecodeError, KeyError):
        return [{"error": "Failed to generate learning path"}]

async def create_enhanced_learning_step(learning_step, file_content, difficulty_level="intermediate"):
    """
    Creates enhanced educational content for a learning step
    """
    prompt = f"""
    Create enhanced educational content for this learning step:
    
    Title: {learning_step['title']}
    Description: {learning_step['description']}
    Files: {learning_step['files']}
    Difficulty: {difficulty_level}
    
    File Content (first 3000 chars): {file_content[:3000]}...
    
    For coding_exercises, create fill-in-the-blank code snippets in markdown format following exactly the content of the file, this is a example:
    ```python
    # Exercise 1: Complete this code
    import torch
    class LinearModel(nn.Module):
        def __init__(self):
            super(__, self).__init__()
            self.linear = nn.__(1, 1)  # Fill the blanks
    ```
    
    Return a JSON object with:
    {{
        "concept_explanation": "Clear explanation of the main concepts",
        "examples": ["example1", "example2", "example3"],
        "job_market_relevance": "How this skill applies in real jobs with specific examples",
        "coding_exercises": ["```python\\n# Exercise 1\\ncode with blanks\\n```", "```python\\n# Exercise 2\\ncode with blanks\\n```"]
    }}
    """
    
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"error": "Failed to generate enhanced content"}

def generate_exercise_solutions(enhanced_learning_path):
    """
    Generates solutions for fill-in-the-blank coding exercises
    """
    
    for i, step in enumerate(enhanced_learning_path):
        exercise =  step['coding_exercises']
        print(exercise)
        prompt = f"""
        Fill in the blanks (__) in this coding exercise with the correct code:
        
        {exercise}
        
        Return only the completed code with all blanks filled in correctly.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        enhanced_learning_path[i]['solutions'] = response.choices[0].message.content
    
    return enhanced_learning_path


