import modal

app = modal.App(name="github-learning-path-generator")

@app.function(
    image = modal.Image.debian_slim().pip_install([
        "python-fasthtml",
        "openai",
        "neo4j",
        "PyGithub",
        "requests",
    ]),
    secrets=[modal.Secret.from_name("")],
)


from fasthtml.common import *


# import from the main_process.py file by getting the path of the file
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_process import main

app = FastHTML()

@app.route("/")
def home():
    return Html(
        Head(Title("GitHub Learning Path Generator")),
        Body(
            H1("🎓 GitHub Learning Path Generator"),
            Form(
                Div(
                    Label("GitHub URL:", For="github_url"),
                    Input(type="text", name="github_url", placeholder="https://github.com/user/repo", required=True)
                ),
                Div(
                    Label("Number of files to process:", For="file_count"), 
                    Input(type="number", name="file_count", value="20", min="1", max="100")
                ),
                Button("Generate Learning Path", type="submit"),
                method="post", action="/generate"
            )
        )
    )
@app.route("/generate", methods=["POST"])
async def generate(github_url: str, file_count: int):
    try:
        # Process the GitHub repository
        enhanced_learning_path_solutions = await main(github_url, file_count)
        
        return render_learning_path(enhanced_learning_path_solutions, github_url)
    
    except Exception as e:
        return render_error(str(e))
def render_learning_path(enhanced_path, repo_name):
    return Html(
        Head(
            Title(f"Learning Path - {repo_name}"),
            Style("""
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                .step { border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 8px; }
                .step-header { background: #f5f5f5; padding: 10px; margin: -20px -20px 15px -20px; }
                .exercise { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }
                pre { background: #f4f4f4; padding: 10px; overflow-x: auto; }
            """)
        ),
        Body(
            H1(f"🎓 Learning Path for {repo_name}"),
            *[render_step(step) for step in enhanced_path],
            A("← Back to Home", href="/", style="margin-top: 30px; display: inline-block;")
        )
    )

def render_step(step):
    return Div(
        Div(
            H2(f"Step {step['step']}: {step['title']}"),
            P(step['description']),
            cls="step-header"
        ),
        H3("📖 Concept Explanation"),
        P(step['concept_explanation']),
        
        H3("💡 Examples"),
        Ul(*[Li(example) for example in step['examples']]),
        
        H3("💼 Job Market Relevance"),
        P(step['job_market_relevance']),
        
        H3("🏋️ Coding Exercises"),
        *[Div(Pre(exercise), cls="exercise") for exercise in step['coding_exercises']],
        
        cls="step"
    )
def render_error(error_msg):
    return Html(
        Head(Title("Error")),
        Body(
            H1("❌ Error"),
            P(f"Something went wrong: {error_msg}"),
            A("← Back to Home", href="/")
        )
    )
if __name__ == "__main__":
    serve()
