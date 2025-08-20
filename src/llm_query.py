import json
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_file_content(file_content, file_type, file_name):
    prompt = f"""
    Analyze this {file_type} file and extract the main topics and concepts covered.
    File name: {file_name}
    
    Content: {file_content[:2000]}...
    
    Return a JSON object with a "topics" array: {{"topics": ["topic1", "topic2"]}}
    """
    
    
    response = client.chat.completions.create(
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
            step_number = step.get('step', 'Unknown')
            title = step.get('title', 'Untitled')
            print(f"Generating solutions for Step {step_number}: {title}")
            step['solutions'] = generate_exercise_solutions(step['coding_exercises'])
    return enhanced_learning_path


