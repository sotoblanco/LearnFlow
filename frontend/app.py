from fasthtml.common import *
import sqlite3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import create_learning_path_from_github
from src.learning_generation import create_complete_enhanced_learning_path
from src.llm_query import generate_all_solutions
from src.database import save_learning_path, get_learning_step, check_existing_repository   

# import modal
# app = modal.App("codelap-edu-frontend")

# @app.function(
#     image=modal.Image.debian_slim(python_version="3.12").pip_install(
#         "python-fasthtml==0.5.2"
#     )
# )
# @modal.asgi_app()
def serve():
    app, rt = fast_app()

    @rt("/", methods=["POST"])
    def post(github_url: str, file_count: int, action: str = None):
        try:
            if not github_url.startswith("https://github.com/"):
                return Div("Please enter a valid GitHub URL", style="color: red")
            
            if file_count < 1:
                return Div("Please enter a valid number of files to process", style="color: red")
            
            # Check if repository already exists
            existing_path_id = check_existing_repository(github_url)
            
            if existing_path_id and action != "regenerate":
                if action == "existing":
                    # Repository exists, retrieve existing learning path
                    conn = sqlite3.connect(os.getenv('DB_PATH', 'learning_paths.db'))
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        SELECT repository, total_files FROM learning_paths WHERE id = ?
                    ''', (existing_path_id,))
                    repo_info = cursor.fetchone()
                    
                    cursor.execute('''
                        SELECT id, step_number, title, description FROM learning_steps 
                        WHERE path_id = ? ORDER BY step_number
                    ''', (existing_path_id,))
                    steps = cursor.fetchall()
                    conn.close()
                    
                    # Display cached results
                    learning_steps = []
                    for step in steps:
                        step_id, step_number, title, description = step
                        learning_steps.append(
                            A(
                                Div(
                                    H3(f"Step {step_number}: {title}"),
                                    P(description),
                                    style="border: 1px solid #ccc; padding: 10px; margin: 10px 0"
                                ),
                                href=f"/step/{step_id}",
                                style="text-decoration: none; color: inherit"
                            )
                        )
                    
                    return Div(
                        H2(f"Learning Path for {repo_info[0]} (Cached)"),
                        P(f"Generated from {repo_info[1]} files"),
                        *learning_steps
                    )
                else:
                    # Get repository info including file count
                    conn = sqlite3.connect(os.getenv('DB_PATH', 'learning_paths.db'))
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT repository, total_files FROM learning_paths WHERE id = ?
                    ''', (existing_path_id,))
                    repo_info = cursor.fetchone()
                    conn.close()

                    return Div(
                        H2("Repository Already Processed"),
                        P(f"This repository has already been processed with {repo_info[1]} files. What would you like to do?"),
                        Form(
                        Hidden(name="github_url", value=github_url),
                        Hidden(name="file_count", value=file_count),
                        Button("Use Existing", name="action", value="existing"),
                        Button("Regenerate", name="action", value="regenerate"),
                        method="post"
                        )
                    )

        
        
            # Repository doesn't exist, process as normal
            result = create_learning_path_from_github(github_url, file_count)
            
            # Check for errors BEFORE processing
            if "error" in result:
                return Div(f"Error: {result['error']}", style="color: red")
            
            # Only process if no errors
            enhanced_learning_path = create_complete_enhanced_learning_path(result)
            complete_path_with_solutions = generate_all_solutions(enhanced_learning_path)
            path_id, step_ids = save_learning_path(result, complete_path_with_solutions, github_url)
            
            # Display results with clickable steps
            learning_steps = []
            for i, step in enumerate(result['learning_path']):
                step_id = step_ids[i]
                learning_steps.append(
                    A(
                        Div(
                            H3(f"Step {step['step']}: {step['title']}"),
                            P(step['description']),
                            style="border: 1px solid #ccc; padding: 10px; margin: 10px 0"
                        ),
                        href=f"/step/{step_id}",
                        style="text-decoration: none; color: inherit"
                    )
                )
            
            return Div(
                H2(f"Learning Path for {result['repository']}"),
                P(f"Generated from {result['total_files']} files"),
                *learning_steps
            )
        
        except Exception as e:
            return Div(f"Error processing repository: {str(e)}", style="color: red")

    @rt("/", methods=["GET"])
    def get():
        return Titled("GraphRAG Learning Path Generator",
                    H1("GitHub Repository to Learning Path"),
                    Form(Input(placeholder="Enter GitHub URL", name="github_url"),
                    Input(type="number", value="15", name="file_count", placeholder="Number of files to process"),
                        Button("Generate Learning Path"),
                        method="post"))

    @rt("/step/{step_id}", methods=["GET"])
    def get_step_detail(step_id: int):
        step = get_learning_step(step_id)
        
        if not step:
            return Div("Step not found", style="color: red")
        
        return Titled(f"Step {step['step']}: {step['title']}",
            H1(step['title']),
            P(step['description']),
            
            H2("Concept Explanation"),
            P(step['concept_explanation']),
            
            H2("Examples"),
            Ul(*[Li(example) for example in step['examples']]),
            
            H2("Job Market Relevance"),
            P(step['job_market_relevance']),
            
            H2("Coding Exercises"),
            *[Div(Pre(exercise), style="background: #f5f5f5; padding: 10px; margin: 10px 0") 
            for exercise in step['coding_exercises']],
            
            H2("Solutions"),
            *[Div(Pre(solution), style="background: #e8f5e8; padding: 10px; margin: 10px 0") 
            for solution in step['solutions']],
            
            A("← Back to Learning Path", href="/", style="margin-top: 20px; display: block")
        )

    return app

if __name__ == "__main__":
    serve()