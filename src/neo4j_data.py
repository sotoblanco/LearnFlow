from neo4j import GraphDatabase
from src.llm_query import analyze_file_content_async
import os
import asyncio  

# Get Neo4j credentials from environment variables
uri = os.getenv('NEO4J_URI', "neo4j+s://cb1b78cd.databases.neo4j.io")
username = os.getenv('NEO4J_USERNAME', "neo4j")
password = os.getenv('NEO4J_PASSWORD')

if not password:
    raise ValueError("NEO4J_PASSWORD environment variable is required")


driver = GraphDatabase.driver(uri, auth=(username, password))

def clear_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def create_schema():
    with driver.session() as session:
        # Create constraints and indexes
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Repository) REQUIRE r.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")

def create_repository_node_from_repo(repo):
    with driver.session() as session:
        session.run("""
            CREATE (r:Repository {
                name: $name,
                url: $url,
                description: $desc
            })
        """, 
        name=repo.full_name, 
        url=repo.html_url, 
        desc=repo.description or "No description")

def add_files_to_graph_with_content(files_data, repo_name, batch_size=10):
    with driver.session() as session:
        count = 0
        for file_path, file_info in files_data.items():
            if count >= batch_size:
                break
            
            session.run("""
                MATCH (r:Repository {name: $repo_name})
                CREATE (f:File {
                    path: $path,
                    name: $name,
                    size: $size,
                    type: $type,
                    large_file: $large_file,
                    content: $content
                })
                CREATE (r)-[:CONTAINS]->(f)
            """, 
            repo_name=repo_name,
            path=file_path,
            name=file_info['content_file'].name,
            size=file_info['size'],
            type=file_path.split('.')[-1],
            large_file=file_info['large_file'],
            content=file_info['content'])
            count += 1
        
        print(f"Added {count} files with content to the graph")

async def process_all_files_async():
    with driver.session() as session:
        result = session.run("MATCH (f:File) RETURN f.name, f.content, f.type")
        files = [record for record in result]
        
    tasks = []
    for file_record in files:
        file_name = file_record['f.name']
        file_content = file_record['f.content']
        file_type = file_record['f.type']
        
        print(f"Creating task for {file_name}...")
        task = analyze_file_content_async(file_content, file_type, file_name)
        tasks.append(task)
    
    print("Processing all files concurrently...")
    results = await asyncio.gather(*tasks)
    
    # Now process the results and add to graph
    for i, topics in enumerate(results):
        file_name = files[i]['f.name']
        print(f"Adding topics for {file_name}...")
        
        add_topics_to_graph(topics)
        connect_file_to_topics(file_name, topics)
    
    print("All files processed!")

def process_all_files():
    with driver.session() as session:
        # Get all files with content
        result = session.run("MATCH (f:File) RETURN f.name, f.content, f.type")
        files = [record for record in result]
    
    for file_record in files:
        file_name = file_record['f.name']
        file_content = file_record['f.content']
        file_type = file_record['f.type']
        
        print(f"Processing {file_name}...")
        
        # Extract topics
        topics = analyze_file_content(file_content, file_type, file_name)
        
        # Add topics to graph
        add_topics_to_graph(topics)
        
        # Connect file to topics
        connect_file_to_topics(file_name, topics)
        
    print("All files processed!")

def get_file_with_content(file_name):
    with driver.session() as session:
        result = session.run("""
            MATCH (f:File {name: $name})
            RETURN f.content, f.type, f.name
        """, name=file_name)
        return result.single()

def add_topics_to_graph(topics_list):
    with driver.session() as session:
        for topic in topics_list:
            session.run("""
                MERGE (t:Topic {name: $topic_name})
            """, topic_name=topic)
        print(f"Added {len(topics_list)} topics")

def connect_file_to_topics(file_name, topics_list):
    with driver.session() as session:
        for topic in topics_list:
            session.run("""
                MATCH (f:File {name: $file_name})
                MATCH (t:Topic {name: $topic_name})
                MERGE (f)-[:COVERS]->(t)
            """, file_name=file_name, topic_name=topic)
        print(f"Connected {file_name} to {len(topics_list)} topics")

def query_files_by_type(file_type):
    with driver.session() as session:
        result = session.run("""
            MATCH (r:Repository)-[:CONTAINS]->(f:File)
            WHERE f.type = $type
            RETURN f.name, f.path, f.size
        """, type=file_type)
        return [record for record in result]







