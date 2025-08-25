from fasthtml.common import *
import asyncio
import json

import os

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.github_data import get_quick_repo_info
from main_process import main, create_learning_path_from_github, create_complete_enhanced_learning_path, generate_exercise_solutions, save_learning_path
app = FastHTML()

import urllib.parse 

import uuid
import threading
from typing import Dict, Any

# Global dictionary to store processing status
processing_tasks: Dict[str, Dict[str, Any]] = {}


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
        # Get quick repo info first
        repo_info = get_quick_repo_info(github_url)
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        processing_tasks[task_id] = {
            "status": "starting",
            "message": "Initializing processing...",
            "progress": 0
        }
        
        # Start background processing
        def background_process():
            try:
                processing_tasks[task_id]["status"] = "processing"
                processing_tasks[task_id]["message"] = "Setting up database..."
                
                # Run the async main function in the background thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                enhanced_learning_path_solutions = loop.run_until_complete(
                    main_with_progress_sync(github_url, file_count, task_id)
                )
                
                processing_tasks[task_id] = {
                    "status": "complete",
                    "message": "Learning path generated successfully!",
                    "data": enhanced_learning_path_solutions
                }
                
            except Exception as e:
                processing_tasks[task_id] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Start background thread
        thread = threading.Thread(target=background_process, daemon=True)
        thread.start()
        
        # Return processing page with task ID
        return render_processing_page_polling(repo_info, file_count, task_id)
        
    except Exception as e:
        return render_error(str(e))

@app.route("/check-status/{task_id}")
def check_status(task_id: str):
    """API endpoint to check processing status"""
    status = processing_tasks.get(task_id, {"status": "not_found", "message": "Task not found"})
    return status

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

def render_repo_info_page(repo_info, file_count):
    return Html(
        Head(
            Title(f"Processing {repo_info['name']}"),
            Style("""
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                .repo-header { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .topics { display: flex; flex-wrap: wrap; gap: 5px; }
                .topic { background: #007bff; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px; }
                .processing { text-align: center; margin: 30px 0; }
                .spinner { /* your spinner CSS */ }
            """)
        ),
        Body(
            H1("🎓 Learning Path Generator"),
            Div(
                H2(f"📚 {repo_info['name']}"),
                P(repo_info['description']),
                P(f"⭐ {repo_info['stars']} stars | 🔤 {repo_info['language']}"),
                Div([Span(topic, cls="topic") for topic in repo_info['topics'][:10]], cls="topics"),
                cls="repo-header"
            ),
            
            Div(
                H3("🔄 Processing Learning Path..."),
                P(f"Analyzing {file_count} files to create your personalized learning path"),
                Div(cls="spinner"),
                cls="processing"
            )
        )
    )

def render_processing_page(repo_info, file_count, github_url):
    # URL encode the github_url in Python
    encoded_github_url = urllib.parse.quote(github_url, safe='')
    
    return Html(
        Head(
            Title(f"Processing {repo_info['name']}"),
            Style("""
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                .repo-header { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .topics { display: flex; flex-wrap: wrap; gap: 5px; }
                .topic { background: #007bff; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px; }
                .processing { text-align: center; margin: 30px 0; }
                .progress { margin: 20px 0; }
                .hidden { display: none; }
                .step { border: 1px solid #ddd; margin: 20px 0; border-radius: 8px; overflow: hidden; }
                .step-header { 
                    background: #f5f5f5; 
                    padding: 15px 20px; 
                    cursor: pointer; 
                    user-select: none;
                    transition: background-color 0.2s ease;
                    border-bottom: 1px solid #ddd;
                }
                .step-header:hover { 
                    background: #e9ecef; 
                }
                .step-header h2 { 
                    margin: 0; 
                    display: flex; 
                    justify-content: space-between; 
                    align-items: center; 
                }
                .step-toggle { 
                    font-size: 1.2em; 
                    transition: transform 0.2s ease; 
                }
                .step-toggle.expanded { 
                    transform: rotate(90deg); 
                }
                .step-content { 
                    padding: 20px; 
                    display: none; 
                }
                .step-content.expanded { 
                    display: block; 
                }
                
                /* Enhanced Exercise Styling */
                .exercise { 
                    background: #1e1e1e; 
                    border: 1px solid #333; 
                    border-radius: 8px; 
                    margin: 15px 0; 
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .exercise-header {
                    background: #2d2d2d;
                    padding: 10px 15px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    border-bottom: 1px solid #333;
                    cursor: pointer;
                    user-select: none;
                }
                .exercise-header:hover {
                    background: #3d3d3d;
                }
                .exercise-title {
                    color: #fff;
                    font-weight: bold;
                    font-size: 14px;
                    margin: 0;
                }
                .exercise-controls {
                    display: flex;
                    gap: 10px;
                }
                .exercise-btn {
                    background: #007bff;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    transition: background 0.2s;
                }
                .exercise-btn:hover {
                    background: #0056b3;
                }
                .copy-btn {
                    background: #28a745;
                }
                .copy-btn:hover {
                    background: #1e7e34;
                }
                .copy-btn.copied {
                    background: #17a2b8;
                }
                .exercise-content {
                    display: none;
                    padding: 0;
                }
                .exercise-content.expanded {
                    display: block;
                }
                .exercise-section {
                    padding: 15px;
                    border-bottom: 1px solid #333;
                }
                .exercise-code {
                    margin: 0 !important;
                    border-radius: 4px !important;
                    background: #1e1e1e !important;
                    color: #f8f8f2;
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                    font-size: 14px;
                    line-height: 1.5;
                    padding: 15px !important;
                    overflow-x: auto;
                    white-space: pre;
                }
                
                /* Solution Styling */
                .solution-container {
                    background: #2d4a22;
                    margin: 0;
                    border-top: 1px solid #4a7c59;
                }
                .solution-header {
                    background: #3d5a32;
                    padding: 10px 15px;
                    cursor: pointer;
                    user-select: none;
                    border-bottom: 1px solid #4a7c59;
                }
                .solution-header:hover {
                    background: #4d6a42;
                }
                .solution-title {
                    margin: 0;
                    color: #90ee90;
                    font-size: 14px;
                    font-weight: bold;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .solution-controls {
                    display: flex;
                    gap: 10px;
                    align-items: center;
                }
                .solution-btn {
                    background: #28a745;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    transition: background 0.2s;
                }
                .solution-btn:hover {
                    background: #1e7e34;
                }
                .solution-toggle {
                    background: #17a2b8;
                    color: white;
                    padding: 5px 10px;
                    border-radius: 4px;
                    font-size: 12px;
                    cursor: pointer;
                    transition: background 0.2s;
                }
                .solution-toggle:hover {
                    background: #138496;
                }
                .solution-content {
                    display: none;
                    padding: 15px;
                    background: #1a3d1a;
                }
                .solution-content.expanded {
                    display: block;
                }
                .solution-code {
                    margin: 0 !important;
                    border-radius: 4px !important;
                    background: #0d2818 !important;
                    color: #90ee90;
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                    font-size: 14px;
                    line-height: 1.5;
                    padding: 15px !important;
                    overflow-x: auto;
                    white-space: pre;
                    border: 1px solid #4a7c59;
                }
                
                .progress-bar { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; }
                .progress-fill { height: 100%; background: #007bff; transition: width 0.3s ease; }
                .step-number { 
                    background: #007bff; 
                    color: white; 
                    border-radius: 50%; 
                    width: 30px; 
                    height: 30px; 
                    display: inline-flex; 
                    align-items: center; 
                    justify-content: center; 
                    margin-right: 10px; 
                    font-weight: bold; 
                }
                .learning-path-header {
                    text-align: center;
                    margin: 30px 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 10px;
                }
                .expand-all-btn {
                    background: #28a745;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    margin: 10px 5px;
                    font-size: 14px;
                }
                .expand-all-btn:hover {
                    background: #218838;
                }
                .collapse-all-btn {
                    background: #6c757d;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    margin: 10px 5px;
                    font-size: 14px;
                }
                .collapse-all-btn:hover {
                    background: #5a6268;
                }
                
                /* Toast notification for copy feedback */
                .toast {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #28a745;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px;
                    z-index: 1000;
                    opacity: 0;
                    transition: opacity 0.3s;
                }
                .toast.show {
                    opacity: 1;
                }
            """)
        ),
        Body(
            # Toast notification element
            Div(id="toast", cls="toast"),
            
            H1("🎓 Learning Path Generator"),
            Div(
                H2(f"📚 {repo_info['name']}"),
                P(repo_info['description']),
                P(f"⭐ {repo_info['stars']} stars | 🔤 {repo_info['language']}"),
                Div([Span(topic, cls="topic") for topic in repo_info['topics'][:10]], cls="topics"),
                cls="repo-header"
            ),
            
            Div(
                H3("🔄 Processing Learning Path...", id="status-title"),
                P(f"Analyzing {file_count} files to create your personalized learning path", id="status-message"),
                Div(
                    Div(id="progress-fill", cls="progress-fill", style="width: 0%"),
                    cls="progress-bar"
                ),
                Div(id="learning-path-container", cls="hidden"),
                cls="processing"
            ),
            
            Script(f"""
                const eventSource = new EventSource('/process-stream?github_url={encoded_github_url}&file_count={file_count}');
                
                eventSource.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    
                    if (data.status === 'starting') {{
                        document.getElementById('status-message').textContent = data.message;
                    }} else if (data.status === 'progress') {{
                        document.getElementById('status-message').textContent = data.message;
                    }} else if (data.status === 'complete') {{
                        document.getElementById('status-title').textContent = '✅ Learning Path Complete!';
                        document.getElementById('status-message').textContent = 'Your personalized learning path is ready.';
                        
                        // Render the learning path
                        const container = document.getElementById('learning-path-container');
                        container.innerHTML = renderLearningPath(data.data);
                        container.classList.remove('hidden');
                        
                        eventSource.close();
                    }} else if (data.status === 'error') {{
                        document.getElementById('status-title').textContent = '❌ Error';
                        document.getElementById('status-message').textContent = data.message;
                        eventSource.close();
                    }}
                }};
                
                function renderLearningPath(learningPath) {{
                    // Convert learning path data to HTML using the same structure as render_step
                    let html = '';
                    learningPath.learning_path.forEach(step => {{
                        html += `
                            <div class="step">
                                <div class="step-header">
                                    <h2>Step ${{step.step}}: ${{step.title}}</h2>
                                    <p>${{step.description}}</p>
                                </div>
                                <h3>📖 Concept Explanation</h3>
                                <p>${{step.concept_explanation}}</p>
                                
                                <h3>💡 Examples</h3>
                                <ul>${{step.examples.map(ex => `<li>${{ex}}</li>`).join('')}}</ul>
                                
                                <h3>💼 Job Market Relevance</h3>
                                <p>${{step.job_market_relevance}}</p>
                                
                                <h3>🏋️ Coding Exercises</h3>
                                ${{step.coding_exercises.map(exercise => `<div class="exercise"><pre>${{exercise}}</pre></div>`).join('')}}
                            </div>
                        `;
                    }});
                    html += '<a href="/" style="margin-top: 30px; display: inline-block;">← Back to Home</a>';
                    return html;
                }}
            """)
        )
    )

async def main_with_progress(github_url, file_count, progress_callback):
    """Modified main function that sends progress updates"""
    
    await progress_callback(f"data: {json.dumps({'status': 'progress', 'message': 'Setting up database...'})}\n\n")
    
    # Your existing main logic with progress updates
    learning_path = await create_learning_path_from_github(github_url, file_count)
    
    await progress_callback(f"data: {json.dumps({'status': 'progress', 'message': 'Generating enhanced learning path...'})}\n\n")
    
    enhanced_learning_path = await create_complete_enhanced_learning_path(learning_path)
    enhanced_learning_path_solutions = generate_exercise_solutions(enhanced_learning_path)
    
    await progress_callback(f"data: {json.dumps({'status': 'progress', 'message': 'Saving to database...'})}\n\n")
    
    path_id, step_ids = save_learning_path(learning_path, enhanced_learning_path_solutions, github_url)
    
    return {
        "repository": github_url,
        "total_files": file_count,
        "learning_path": enhanced_learning_path_solutions
    }

async def main_with_progress_sync(github_url, file_count, task_id):
    """Modified main function that updates global status"""
    
    processing_tasks[task_id]["message"] = "Setting up database..."
    processing_tasks[task_id]["progress"] = 10
    
    learning_path = await create_learning_path_from_github(github_url, file_count)
    
    processing_tasks[task_id]["message"] = "Generating enhanced learning path..."
    processing_tasks[task_id]["progress"] = 60
    
    enhanced_learning_path = await create_complete_enhanced_learning_path(learning_path)
    enhanced_learning_path_solutions = generate_exercise_solutions(enhanced_learning_path)
    
    processing_tasks[task_id]["message"] = "Saving to database..."
    processing_tasks[task_id]["progress"] = 90
    
    path_id, step_ids = save_learning_path(learning_path, enhanced_learning_path_solutions, github_url)
    
    return {
        "repository": github_url,
        "total_files": file_count,
        "learning_path": enhanced_learning_path_solutions
    }

def render_processing_page_polling(repo_info, file_count, task_id):
    return Html(
        Head(
            Title(f"Processing {repo_info['name']}"),
            # Add Prism.js for syntax highlighting
            Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css"),
            Script(src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"),
            Script(src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"),
            Style("""
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                .repo-header { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .topics { display: flex; flex-wrap: wrap; gap: 5px; }
                .topic { background: #007bff; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px; }
                .processing { text-align: center; margin: 30px 0; }
                .progress { margin: 20px 0; }
                .hidden { display: none; }
                .step { border: 1px solid #ddd; margin: 20px 0; border-radius: 8px; overflow: hidden; }
                .step-header { 
                    background: #f5f5f5; 
                    padding: 15px 20px; 
                    cursor: pointer; 
                    user-select: none;
                    transition: background-color 0.2s ease;
                    border-bottom: 1px solid #ddd;
                }
                .step-header:hover { 
                    background: #e9ecef; 
                }
                .step-header h2 { 
                    margin: 0; 
                    display: flex; 
                    justify-content: space-between; 
                    align-items: center; 
                }
                .step-toggle { 
                    font-size: 1.2em; 
                    transition: transform 0.2s ease; 
                }
                .step-toggle.expanded { 
                    transform: rotate(90deg); 
                }
                .step-content { 
                    padding: 20px; 
                    display: none; 
                }
                .step-content.expanded { 
                    display: block; 
                }
                
                /* Enhanced Exercise Styling */
                .exercise { 
                    background: #1e1e1e; 
                    border: 1px solid #333; 
                    border-radius: 8px; 
                    margin: 15px 0; 
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .exercise-header {
                    background: #2d2d2d;
                    padding: 10px 15px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    border-bottom: 1px solid #333;
                    cursor: pointer;
                    user-select: none;
                }
                .exercise-header:hover {
                    background: #3d3d3d;
                }
                .exercise-title {
                    color: #fff;
                    font-weight: bold;
                    font-size: 14px;
                    margin: 0;
                }
                .exercise-controls {
                    display: flex;
                    gap: 10px;
                }
                .exercise-btn {
                    background: #007bff;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    transition: background 0.2s;
                }
                .exercise-btn:hover {
                    background: #0056b3;
                }
                .copy-btn {
                    background: #28a745;
                }
                .copy-btn:hover {
                    background: #1e7e34;
                }
                .copy-btn.copied {
                    background: #17a2b8;
                }
                .exercise-content {
                    display: none;
                    padding: 0;
                }
                .exercise-content.expanded {
                    display: block;
                }
                .exercise-section {
                    padding: 15px;
                    border-bottom: 1px solid #333;
                }
                .exercise-code {
                    margin: 0 !important;
                    border-radius: 4px !important;
                    background: #1e1e1e !important;
                    color: #f8f8f2;
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                    font-size: 14px;
                    line-height: 1.5;
                    padding: 15px !important;
                    overflow-x: auto;
                    white-space: pre;
                }
                
                /* Solution Styling */
                .solution-container {
                    background: #2d4a22;
                    margin: 0;
                    border-top: 1px solid #4a7c59;
                }
                .solution-header {
                    background: #3d5a32;
                    padding: 10px 15px;
                    cursor: pointer;
                    user-select: none;
                    border-bottom: 1px solid #4a7c59;
                }
                .solution-header:hover {
                    background: #4d6a42;
                }
                .solution-title {
                    margin: 0;
                    color: #90ee90;
                    font-size: 14px;
                    font-weight: bold;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .solution-controls {
                    display: flex;
                    gap: 10px;
                    align-items: center;
                }
                .solution-btn {
                    background: #28a745;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    transition: background 0.2s;
                }
                .solution-btn:hover {
                    background: #1e7e34;
                }
                .solution-toggle {
                    background: #17a2b8;
                    color: white;
                    padding: 5px 10px;
                    border-radius: 4px;
                    font-size: 12px;
                    cursor: pointer;
                    transition: background 0.2s;
                }
                .solution-toggle:hover {
                    background: #138496;
                }
                .solution-content {
                    display: none;
                    padding: 15px;
                    background: #1a3d1a;
                }
                .solution-content.expanded {
                    display: block;
                }
                .solution-code {
                    margin: 0 !important;
                    border-radius: 4px !important;
                    background: #0d2818 !important;
                    color: #90ee90;
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                    font-size: 14px;
                    line-height: 1.5;
                    padding: 15px !important;
                    overflow-x: auto;
                    white-space: pre;
                    border: 1px solid #4a7c59;
                }
                
                .progress-bar { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; }
                .progress-fill { height: 100%; background: #007bff; transition: width 0.3s ease; }
                .step-number { 
                    background: #007bff; 
                    color: white; 
                    border-radius: 50%; 
                    width: 30px; 
                    height: 30px; 
                    display: inline-flex; 
                    align-items: center; 
                    justify-content: center; 
                    margin-right: 10px; 
                    font-weight: bold; 
                }
                .learning-path-header {
                    text-align: center;
                    margin: 30px 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 10px;
                }
                .expand-all-btn {
                    background: #28a745;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    margin: 10px 5px;
                    font-size: 14px;
                }
                .expand-all-btn:hover {
                    background: #218838;
                }
                .collapse-all-btn {
                    background: #6c757d;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    margin: 10px 5px;
                    font-size: 14px;
                }
                .collapse-all-btn:hover {
                    background: #5a6268;
                }
                
                /* Toast notification for copy feedback */
                .toast {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #28a745;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px;
                    z-index: 1000;
                    opacity: 0;
                    transition: opacity 0.3s;
                }
                .toast.show {
                    opacity: 1;
                }
            """)
        ),
        Body(
            # Toast notification element
            Div(id="toast", cls="toast"),
            
            H1("🎓 Learning Path Generator"),
            Div(
                H2(f"📚 {repo_info['name']}"),
                P(repo_info['description']),
                P(f"⭐ {repo_info['stars']} stars | 🔤 {repo_info['language']}"),
                Div([Span(topic, cls="topic") for topic in repo_info['topics'][:10]], cls="topics"),
                cls="repo-header"
            ),
            
            Div(
                H3("🔄 Processing Learning Path...", id="status-title"),
                P(f"Analyzing {file_count} files to create your personalized learning path", id="status-message"),
                Div(
                    Div(id="progress-fill", cls="progress-fill", style="width: 0%"),
                    cls="progress-bar"
                ),
                Div(id="learning-path-container", cls="hidden"),
                cls="processing"
            ),
            
            Script(f"""
                let taskId = '{task_id}';
                
                function checkStatus() {{
                    fetch(`/check-status/${{taskId}}`)
                        .then(response => response.json())
                        .then(data => {{
                            if (data.status === 'complete') {{
                                document.getElementById('status-title').textContent = '✅ Learning Path Complete!';
                                document.getElementById('status-message').textContent = 'Your personalized learning path is ready.';
                                document.getElementById('progress-fill').style.width = '100%';
                                
                                // Render the learning path
                                const container = document.getElementById('learning-path-container');
                                container.innerHTML = renderLearningPath(data.data);
                                container.classList.remove('hidden');
                                
                                // Add click event listeners for steps
                                addStepInteractivity();
                                
                            }} else if (data.status === 'error') {{
                                document.getElementById('status-title').textContent = '❌ Error';
                                document.getElementById('status-message').textContent = data.message;
                                
                            }} else if (data.status === 'processing' || data.status === 'starting') {{
                                document.getElementById('status-message').textContent = data.message;
                                if (data.progress) {{
                                    document.getElementById('progress-fill').style.width = data.progress + '%';
                                }}
                                // Continue polling
                                setTimeout(checkStatus, 2000);
                                
                            }} else {{
                                // Still processing, check again
                                setTimeout(checkStatus, 2000);
                            }}
                        }})
                        .catch(error => {{
                            console.error('Error checking status:', error);
                            setTimeout(checkStatus, 3000); // Retry after error
                        }});
                }}
                
                function renderLearningPath(learningPath) {{
                    let html = `
                        <div class="learning-path-header">
                            <h2>🎯 Your Personalized Learning Path</h2>
                            <p>Click on any step to expand and explore the content</p>
                            <button class="expand-all-btn" onclick="expandAllSteps()">📖 Expand All</button>
                            <button class="collapse-all-btn" onclick="collapseAllSteps()">📚 Collapse All</button>
                        </div>
                    `;
                    
                    learningPath.learning_path.forEach((step, index) => {{
                        html += `
                            <div class="step" data-step="${{step.step}}">
                                <div class="step-header" onclick="toggleStep(${{step.step}})">
                                    <h2>
                                        <span>
                                            <span class="step-number">${{step.step}}</span>
                                            ${{step.title}}
                                        </span>
                                        <span class="step-toggle" id="toggle-${{step.step}}">▶</span>
                                    </h2>
                                </div>
                                <div class="step-content" id="content-${{step.step}}">
                                    <p><strong>📝 Description:</strong> ${{step.description}}</p>
                                    
                                    <h3>📖 Concept Explanation</h3>
                                    <p>${{step.concept_explanation}}</p>
                                    
                                    <h3>💡 Examples</h3>
                                    <ul>${{step.examples.map(ex => `<li>${{ex}}</li>`).join('')}}</ul>
                                    
                                    <h3>💼 Job Market Relevance</h3>
                                    <p>${{step.job_market_relevance}}</p>
                                    
                                    <h3>🏋️ Coding Exercises</h3>
                                    ${{renderExercises(step.coding_exercises, step.step)}}
                                </div>
                            </div>
                        `;
                    }});
                    
                    html += '<a href="/" style="margin-top: 30px; display: inline-block; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">← Back to Home</a>';
                    return html;
                }}
                
                function renderExercises(exercises, stepNumber) {{
                    if (!exercises || exercises.length === 0) {{
                        return '<p>No coding exercises for this step.</p>';
                    }}
                    
                    return exercises.map((exercise, index) => {{
                        const exerciseId = `exercise-${{stepNumber}}-${{index}}`;
                        const solutionId = `solution-${{stepNumber}}-${{index}}`;
                        
                        // Split exercise and solution if they exist
                        const parts = exercise.split('## Solution:');
                        const exerciseCode = cleanCodeContent(parts[0]);
                        const solutionCode = parts.length > 1 ? cleanCodeContent(parts[1]) : '';
                        
                        const language = detectLanguage(exerciseCode);
                        
                        return `
                            <div class="exercise">
                                <div class="exercise-header" onclick="toggleExercise('${{exerciseId}}')">
                                    <h4 class="exercise-title">💻 Exercise ${{index + 1}} (${{language}})</h4>
                                    <div class="exercise-controls">
                                        <button class="exercise-btn copy-btn" onclick="copyCode('${{exerciseId}}', event)" title="Copy exercise code">
                                            📋 Copy Exercise
                                        </button>
                                        <span class="exercise-btn" id="toggle-icon-${{exerciseId}}">▶</span>
                                    </div>
                                </div>
                                <div class="exercise-content" id="${{exerciseId}}">
                                    <div class="exercise-section">
                                        <h5 style="margin: 0 0 10px 0; color: #007bff; font-size: 14px;">📝 Exercise:</h5>
                                        <pre class="exercise-code language-${{language}}" id="code-${{exerciseId}}">${{escapeHtml(exerciseCode)}}</pre>
                                    </div>
                                    ${{solutionCode ? renderSolution(solutionCode, solutionId, language) : ''}}
                                </div>
                            </div>
                        `;
                    }}).join('');
                }}
                
                function renderSolution(solutionCode, solutionId, language) {{
                    return `
                        <div class="solution-container">
                            <div class="solution-header" onclick="toggleSolution('${{solutionId}}')">
                                <h5 class="solution-title">
                                    <span>💡 Solution</span>
                                    <div class="solution-controls">
                                        <button class="solution-btn copy-btn" onclick="copySolution('${{solutionId}}', event)" title="Copy solution">
                                            📋 Copy Solution
                                        </button>
                                        <span class="solution-toggle" id="toggle-icon-${{solutionId}}">👁️ Show</span>
                                    </div>
                                </h5>
                            </div>
                            <div class="solution-content" id="${{solutionId}}">
                                <pre class="solution-code language-${{language}}" id="code-${{solutionId}}">${{escapeHtml(solutionCode)}}</pre>
                            </div>
                        </div>
                    `;
                }}
                
                function cleanCodeContent(code) {{
                    // Remove markdown code block syntax
                    let cleaned = code.replace(/^```[a-zA-Z]*\\n?/, '').replace(/```$/, '');
                    // Remove leading/trailing whitespace but preserve internal formatting
                    cleaned = cleaned.replace(/^\\s+|\\s+$/g, '');
                    return cleaned;
                }}
                
                function detectLanguage(code) {{
                    // Simple language detection based on common patterns
                    if (code.includes('def ') || code.includes('import ') || code.includes('print(')) {{
                        return 'python';
                    }} else if (code.includes('function ') || code.includes('const ') || code.includes('console.log')) {{
                        return 'javascript';
                    }} else if (code.includes('public class') || code.includes('System.out.println')) {{
                        return 'java';
                    }} else if (code.includes('#include') || code.includes('cout <<')) {{
                        return 'cpp';
                    }} else if (code.includes('SELECT') || code.includes('FROM')) {{
                        return 'sql';
                    }}
                    return 'text';
                }}
                
                function escapeHtml(text) {{
                    const div = document.createElement('div');
                    div.textContent = text;
                    return div.innerHTML;
                }}
                
                function toggleExercise(exerciseId) {{
                    const content = document.getElementById(exerciseId);
                    const toggle = document.getElementById(`toggle-icon-${{exerciseId}}`);
                    
                    if (content.classList.contains('expanded')) {{
                        content.classList.remove('expanded');
                        toggle.textContent = '▶';
                    }} else {{
                        content.classList.add('expanded');
                        toggle.textContent = '▼';
                        
                        // Trigger syntax highlighting after content is visible
                        setTimeout(() => {{
                            if (window.Prism) {{
                                Prism.highlightAllUnder(content);
                            }}
                        }}, 100);
                    }}
                }}
                
                function copyCode(exerciseId, event) {{
                    event.stopPropagation(); // Prevent triggering the exercise toggle
                    
                    const codeElement = document.getElementById(`code-${{exerciseId}}`);
                    const code = codeElement.textContent;
                    
                    navigator.clipboard.writeText(code).then(() => {{
                        showToast('Code copied to clipboard!');
                        
                        // Visual feedback on copy button
                        const copyBtn = event.target;
                        const originalText = copyBtn.textContent;
                        copyBtn.textContent = '✅ Copied';
                        copyBtn.classList.add('copied');
                        
                        setTimeout(() => {{
                            copyBtn.textContent = originalText;
                            copyBtn.classList.remove('copied');
                        }}, 2000);
                    }}).catch(err => {{
                        console.error('Failed to copy: ', err);
                        showToast('Failed to copy code', 'error');
                    }});
                }}
                
                function showToast(message, type = 'success') {{
                    const toast = document.getElementById('toast');
                    toast.textContent = message;
                    toast.className = `toast ${{type}}`;
                    toast.classList.add('show');
                    
                    setTimeout(() => {{
                        toast.classList.remove('show');
                    }}, 3000);
                }}
                
                function addStepInteractivity() {{
                    // Auto-expand first step
                    toggleStep(1);
                }}
                
                function toggleStep(stepNumber) {{
                    const content = document.getElementById(`content-${{stepNumber}}`);
                    const toggle = document.getElementById(`toggle-${{stepNumber}}`);
                    
                    if (content.classList.contains('expanded')) {{
                        content.classList.remove('expanded');
                        toggle.classList.remove('expanded');
                        toggle.textContent = '▶';
                    }} else {{
                        content.classList.add('expanded');
                        toggle.classList.add('expanded');
                        toggle.textContent = '▼';
                        
                        // Trigger syntax highlighting for any visible code blocks
                        setTimeout(() => {{
                            if (window.Prism) {{
                                Prism.highlightAllUnder(content);
                            }}
                        }}, 100);
                    }}
                }}
                
                function expandAllSteps() {{
                    const steps = document.querySelectorAll('.step');
                    steps.forEach(step => {{
                        const stepNumber = step.dataset.step;
                        const content = document.getElementById(`content-${{stepNumber}}`);
                        const toggle = document.getElementById(`toggle-${{stepNumber}}`);
                        
                        content.classList.add('expanded');
                        toggle.classList.add('expanded');
                        toggle.textContent = '▼';
                    }});
                    
                    // Trigger syntax highlighting for all visible code
                    setTimeout(() => {{
                        if (window.Prism) {{
                            Prism.highlightAll();
                        }}
                    }}, 100);
                }}
                
                function collapseAllSteps() {{
                    const steps = document.querySelectorAll('.step');
                    steps.forEach(step => {{
                        const stepNumber = step.dataset.step;
                        const content = document.getElementById(`content-${{stepNumber}}`);
                        const toggle = document.getElementById(`toggle-${{stepNumber}}`);
                        
                        content.classList.remove('expanded');
                        toggle.classList.remove('expanded');
                        toggle.textContent = '▶';
                    }});
                }}
                
                function toggleSolution(solutionId) {{
                    const content = document.getElementById(solutionId);
                    const toggle = document.getElementById(`toggle-icon-${{solutionId}}`);
                    
                    if (content.classList.contains('expanded')) {{
                        content.classList.remove('expanded');
                        toggle.textContent = '👁️ Show';
                    }} else {{
                        content.classList.add('expanded');
                        toggle.textContent = '🙈 Hide';
                        
                        // Trigger syntax highlighting after content is visible
                        setTimeout(() => {{
                            if (window.Prism) {{
                                Prism.highlightAllUnder(content);
                            }}
                        }}, 100);
                    }}
                }}
                
                function copySolution(solutionId, event) {{
                    event.stopPropagation(); // Prevent triggering the solution toggle
                    
                    const codeElement = document.getElementById(`code-${{solutionId}}`);
                    const code = codeElement.textContent;
                    
                    navigator.clipboard.writeText(code).then(() => {{
                        showToast('Solution copied to clipboard!');
                        
                        // Visual feedback on copy button
                        const copyBtn = event.target;
                        const originalText = copyBtn.textContent;
                        copyBtn.textContent = '✅ Copied';
                        copyBtn.classList.add('copied');
                        
                        setTimeout(() => {{
                            copyBtn.textContent = originalText;
                            copyBtn.classList.remove('copied');
                        }}, 2000);
                    }}).catch(err => {{
                        console.error('Failed to copy: ', err);
                        showToast('Failed to copy solution', 'error');
                    }});
                }}
                
                // Start checking status
                checkStatus();
            """)
        )
    )

if __name__ == "__main__":
    serve()