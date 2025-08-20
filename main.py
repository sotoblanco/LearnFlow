import os
from github import Github


from src.github_data import get_priority_files
from src.neo4j_data import clear_database, create_schema, create_repository_node_from_repo, add_files_to_graph_with_content, process_all_files
from src.learning_generation import get_learning_data, generate_learning_path

def create_learning_path_from_github(github_url, file_count=100):
    """
    Main function that takes a GitHub URL and returns a learning path
    """
    g = Github(os.getenv('GITHUB_PAT'))
    try:
        # Step 1: Extract repo info from URL
        repo_name = github_url.replace("https://github.com/", "")
        repo = g.get_repo(repo_name)
        
        # Step 2: Clear and setup database
        clear_database()
        create_schema()
        create_repository_node_from_repo(repo)
        
        # Step 3: Ingest files
        content_files = get_priority_files(repo, max_files=file_count)
        add_files_to_graph_with_content(content_files, repo_name, batch_size=len(content_files))
        
        # Step 4: Process topics and relationships
        process_all_files()
        
        # Step 5: Generate learning path
        learning_data = get_learning_data()
        learning_path = generate_learning_path(learning_data)
        
        return {
            "repository": repo_name,
            "total_files": len(content_files),
            "learning_path": learning_path
        }
        
    except Exception as e:
        return {"error": f"Failed to process repository: {str(e)}"}
