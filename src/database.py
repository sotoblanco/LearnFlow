import sqlite3
import json

import sqlite3
import json

def setup_database():
    conn = sqlite3.connect('learning_paths.db')
    cursor = conn.cursor()
    
    # Table for learning paths
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_paths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repository TEXT NOT NULL,
            github_url TEXT NOT NULL,
            total_files INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table for individual steps
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path_id INTEGER,
            step_number INTEGER,
            title TEXT,
            description TEXT,
            files TEXT,
            concept_explanation TEXT,
            examples TEXT,
            job_market_relevance TEXT,
            coding_exercises TEXT,
            solutions TEXT,
            FOREIGN KEY (path_id) REFERENCES learning_paths (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def check_existing_repository(github_url):
    conn = sqlite3.connect('learning_paths.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id FROM learning_paths WHERE github_url = ?
    ''', (github_url,))
    
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result else None

def save_learning_path(result, enhanced_path, github_url):
    conn = sqlite3.connect('learning_paths.db')
    cursor = conn.cursor()
    
    # Save main learning path
    cursor.execute('''
        INSERT INTO learning_paths (repository, github_url, total_files)
        VALUES (?, ?, ?)
    ''', (result['repository'], github_url, result['total_files']))
    
    path_id = cursor.lastrowid
    step_ids = []
    
    # Save individual steps and collect IDs
    for step in enhanced_path:
        cursor.execute('''
            INSERT INTO learning_steps (path_id, step_number, title, description, files,
                                      concept_explanation, examples, job_market_relevance,
                                      coding_exercises, solutions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (path_id, step['step'], step['title'], step['description'],
              json.dumps(step.get('files', [])), step.get('concept_explanation', ''),
              json.dumps(step.get('examples', [])), step.get('job_market_relevance', ''),
              json.dumps(step.get('coding_exercises', [])), json.dumps(step.get('solutions', []))))
        
        step_ids.append(cursor.lastrowid)
    
    conn.commit()
    conn.close()
    return path_id, step_ids

def get_learning_step(step_id):
    conn = sqlite3.connect('learning_paths.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM learning_steps WHERE id = ?
    ''', (step_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    # Convert to dictionary
    step_dict = {
        'id': row[0],
        'path_id': row[1],
        'step': row[2],
        'title': row[3],
        'description': row[4],
        'files': json.loads(row[5]) if row[5] else [],
        'concept_explanation': row[6],
        'examples': json.loads(row[7]) if row[7] else [],
        'job_market_relevance': row[8],
        'coding_exercises': json.loads(row[9]) if row[9] else [],
        'solutions': json.loads(row[10]) if row[10] else []
    }
    
    return step_dict

