import modal

# Create a Modal stub
app = modal.App("codelap-edu")

# Define the image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "neo4j>=5.28.2",
    "openai>=1.100.2", 
    "pygithub>=2.7.0",
    "python-fasthtml>=0.12.24",
    "requests",
    "fastapi",
]).add_local_dir(".", "/root")

secrets=[
    modal.Secret.from_name("github-pat"),
    modal.Secret.from_name("openai-api-key"),
    modal.Secret.from_name("neo4j-credentials")
]
       

# Create a volume for persistent SQLite database
volume = modal.Volume.from_name("codelap-db", create_if_missing=True)

@app.function(image=image, secrets=secrets, volumes={"/data": volume})
@modal.asgi_app()
def web_app():
    """Deploy the existing FastHTML app to Modal"""
    import os
    import sys
    from pathlib import Path

    # Add the entire project root to Python path
    project_root = Path("/root")
    if project_root.exists():
        sys.path.insert(0, str(project_root))
    
    # Set environment variables from secrets
    os.environ["GITHUB_PAT"] = os.environ.get("GITHUB_PAT")
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    os.environ["NEO4J_URI"] = os.environ.get("NEO4J_URI")
    os.environ["NEO4J_USERNAME"] = os.environ.get("NEO4J_USERNAME")
    os.environ["NEO4J_PASSWORD"] = os.environ.get("NEO4J_PASSWORD")
    
    # Initialize database if needed
    try:
        from src.database import setup_database
        setup_database()
    except Exception as e:
        print(f"Database setup warning: {e}")
    
    # Import and run your existing app
    #try:
    from frontend.app_fh import app
    return app


if __name__ == "__main__":
    app.run(web_app)