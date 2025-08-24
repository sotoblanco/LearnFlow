import argparse
import os
from github import Github
import asyncio
from src.github_data import get_priority_files
from src.neo4j_data import clear_database, create_schema, create_repository_node_from_repo, add_files_to_graph_with_content, process_all_files_async
from src.learning_generation import get_learning_data, create_complete_enhanced_learning_path
from src.database import save_learning_path
from src.llm_query import generate_learning_path, generate_exercise_solutions

async def create_learning_path_from_github(github_url, file_count=100):
    """
    Main function that takes a GitHub URL and returns a learning path
    """
    g = Github(os.getenv('GITHUB_PAT'))
    #try:
    # Step 1: Extract repo info from URL
    repo_name = github_url.replace("https://github.com/", "")
    repo = g.get_repo(repo_name)
    
    # Step 2: Clear and setup database
    clear_database()
    create_schema()
    create_repository_node_from_repo(repo)
    
    # Step 3: Ingest files
    content_files = get_priority_files(repo, max_files=file_count)
    add_files_to_graph_with_content(content_files, repo.full_name, batch_size=len(content_files))
    
    # Step 4: Process topics and relationships
    await process_all_files_async()
    
    # Step 5: Generate learning path
    learning_data = get_learning_data()
    learning_path = await generate_learning_path(learning_data)
    
    return {
        "repository": repo_name,
        "total_files": len(content_files),
        "learning_path": learning_path
    }

async def main(github_url, file_count):
    learning_path = await create_learning_path_from_github(github_url, file_count)
    enhanced_learning_path = await create_complete_enhanced_learning_path(learning_path)
    enhanced_learning_path_solutions = generate_exercise_solutions(enhanced_learning_path)
    print(enhanced_learning_path_solutions)
    path_id, step_ids = save_learning_path(learning_path, enhanced_learning_path_solutions, github_url)
    return enhanced_learning_path_solutions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a learning path from a GitHub repository")
    parser.add_argument("--github_url", type=str, required=True, help="The URL of the GitHub repository")
    parser.add_argument("--file_count", type=int, default=100, help="The number of files to process")
    args = parser.parse_args()
    asyncio.run(main(args.github_url, args.file_count))
    
