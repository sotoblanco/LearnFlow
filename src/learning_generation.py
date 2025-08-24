import json
from neo4j import GraphDatabase
from openai import OpenAI
import os
from src.neo4j_data import get_file_with_content
from src.llm_query import create_enhanced_learning_step

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


async def create_complete_enhanced_learning_path(learning_path_result):
    """
    Creates enhanced content for all learning steps
    """
    enhanced_path = []
    
    for step in learning_path_result['learning_path']:
        print(f"Processing step {step['step']}: {step['title']}...")
        
        # Get file content for the first file in the step
        if step['files']:

            file_data = get_file_with_content(step['files'][0])
            file_content = file_data['f.content']
            
            # Create enhanced content
            enhanced_content = await create_enhanced_learning_step(step, file_content)
            
            # Combine original step with enhanced content
            enhanced_step = {**step, **enhanced_content}
            enhanced_path.append(enhanced_step)
            
    return enhanced_path


