import os
from github import Github
import requests

def get_quick_repo_info(github_url):
    repo_name = github_url.replace("https://github.com/", "")
    g = Github(os.getenv('GITHUB_PAT'))
    repo = g.get_repo(repo_name)
    
    return {
        "name": repo.full_name,
        "description": repo.description or "No description available",
        "stars": repo.stargazers_count,
        "language": repo.language,
        "topics": repo.get_topics()
    }

def get_target_files_with_content(repo, path="", target_extensions=['.md', '.ipynb', '.py'], include_content=True):
    content = repo.get_contents(path)
    files = {}
    
    for ct in content:
        if ct.type == "file" and ct.name.endswith(tuple(target_extensions)):
            file_info = {
                'content_file': ct,
                'size': ct.size,
                'large_file': ct.size > 1000000,  # 1MB threshold
                'download_url': ct.download_url
            }
            
            # Add content if requested
            if include_content:
                try:
                    if file_info['large_file']:
                        response = requests.get(file_info['download_url'])
                        file_info['content'] = response.text
                    else:
                        file_info['content'] = ct.decoded_content.decode('utf-8')
                except Exception as e:
                    file_info['content'] = f"Error retrieving content: {str(e)}"
            
            files[ct.path] = file_info
            
        elif ct.type == "dir":
            # Recursively search subdirectories
            subdir_files = get_target_files_with_content(repo, ct.path, target_extensions, include_content)
            files.update(subdir_files)
            
    return files


def get_priority_files(repo, max_files=20):
    all_files = get_target_files_with_content(repo, include_content=False)
    priority_files = {}
    
    # Priority 1: README files (highest priority)
    readme_files = {path: info for path, info in all_files.items() 
                   if 'readme' in path.lower()}
    
    # Priority 2: Tutorial/lesson notebooks
    tutorial_notebooks = {path: info for path, info in all_files.items() 
                         if path.endswith('.ipynb') and 
                         any(keyword in path.lower() for keyword in ['tutorial', 'lesson', 'example', 'demo'])}
    
    # Priority 3: Main Python files (exclude tests, utils)
    main_python = {path: info for path, info in all_files.items() 
                  if path.endswith('.py') and 
                  not any(exclude in path.lower() for exclude in ['test', 'util', '__pycache__', 'setup'])}
    
    # Combine and limit
    priority_files.update(readme_files)
    priority_files.update(list(tutorial_notebooks.items())[:10])  # Max 10 notebooks
    priority_files.update(list(main_python.items())[:5])  # Max 5 Python files
    
    # Now get content for selected files
    final_files = {}
    for path, info in list(priority_files.items())[:max_files]:
        if info['large_file']:
            response = requests.get(info['download_url'])
            info['content'] = response.text
        else:
            info['content'] = info['content_file'].decoded_content.decode('utf-8')
        final_files[path] = info
    
    return final_files

