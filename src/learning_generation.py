import json
from neo4j import GraphDatabase
from openai import OpenAI
import os
from src.neo4j_data import get_file_with_content

# Get Neo4j credentials from environment variables
uri = os.getenv('NEO4J_URI', "neo4j+s://cb1b78cd.databases.neo4j.io")
username = os.getenv('NEO4J_USERNAME', "neo4j")
password = os.getenv('NEO4J_PASSWORD')

if not password:
    raise ValueError("NEO4J_PASSWORD environment variable is required")


driver = GraphDatabase.driver(uri, auth=(username, password))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_learning_data():
    with driver.session() as session:
        result = session.run("""
            MATCH (f:File)-[:COVERS]->(t:Topic)
            RETURN f.name, f.type, collect(t.name) as topics
        """)
        return [record for record in result]

def generate_learning_path(learning_data):
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
    response = client.chat.completions.create(
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

def create_enhanced_learning_step(learning_step, file_content, difficulty_level="intermediate"):
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
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"error": "Failed to generate enhanced content"}

def create_complete_enhanced_learning_path(learning_path_result):
    """
    Creates enhanced content for all learning steps
    """
    enhanced_path = []
    
    for step in learning_path_result['learning_path']:
        print(f"Processing step {step['step']}: {step['title']}...")
        
        # Get file content for the first file in the step
        if step['files']:
            try:
                file_data = get_file_with_content(step['files'][0])
                file_content = file_data['f.content']
                
                # Create enhanced content
                enhanced_content = create_enhanced_learning_step(step, file_content)
                
                # Combine original step with enhanced content
                enhanced_step = {**step, **enhanced_content}
                enhanced_path.append(enhanced_step)
                
            except Exception as e:
                print(f"Error processing {step['files'][0]}: {str(e)}")
                enhanced_path.append(step)  # Keep original if enhancement fails
    
    return enhanced_path

def generate_exercise_solutions(coding_exercises):
    """
    Generates solutions for fill-in-the-blank coding exercises
    """
    solutions = []
    
    for exercise in coding_exercises:
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
        
        solutions.append(response.choices[0].message.content)
    
    return solutions

def generate_all_solutions(enhanced_learning_path):
    """
    Generates solutions for all coding exercises in the enhanced learning path
    """
    for step in enhanced_learning_path:
        if 'coding_exercises' in step:
            print(f"Generating solutions for Step {step['step']}: {step['title']}")
            step['solutions'] = generate_exercise_solutions(step['coding_exercises'])
    
    return enhanced_learning_path


