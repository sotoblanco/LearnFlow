from neo4j import GraphDatabase
import os

# Get Neo4j credentials from environment variables
uri = os.getenv('NEO4J_URI', "neo4j+s://cb1b78cd.databases.neo4j.io")
username = os.getenv('NEO4J_USERNAME', "neo4j")
password = os.getenv('NEO4J_PASSWORD')

if not password:
    raise ValueError("NEO4J_PASSWORD environment variable is required")


def test_connection():
    with driver.session() as session:
        result = session.run("RETURN 'Hello Neo4j!' as message")
        return result.single()["message"]

print(test_connection())
