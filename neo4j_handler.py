# src/core/neo4j_handler.py
from neo4j import GraphDatabase
import uuid
from datetime import datetime
import json
import os
from passlib.hash import pbkdf2_sha256 # For hashing initial admin password
from dotenv import load_dotenv

load_dotenv()

class Neo4jHandler:
    def __init__(self):
        # Load Neo4j connection details from environment variables or .env file
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password") # Replace with your Neo4j password

        # --- DEBUGGING PRINTS ---
        print(f"\n--- Neo4jHandler Initialization Debug ---")
        print(f"NEO4J_URI loaded: {self.uri}")
        print(f"NEO4J_USERNAME loaded: {self.username}")
        # Only print a partial password to avoid exposing it in logs
        print(f"NEO4J_PASSWORD loaded (first 3 chars): {self.password[:3]}...")
        print(f"--- End Debug ---\n")
        # --- END DEBUGGING PRINTS ---

        self.driver = None
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()
            print("Neo4j connection successful!")
            self._initialize_admin_user() # Initialize admin user on successful connection
        except Exception as e:
            print(f"Error connecting to Neo4j: {type(e).__name__}: {e}") # Print exception type and message
            self.driver = None # Ensure driver is None if connection fails

    def _initialize_admin_user(self):
        """Ensures an 'admin' user exists with the 'admin' role."""
        if not self.driver:
            print("Admin user initialization skipped: Neo4j driver not initialized.")
            return False
        admin_username = "admin"
        admin_password = "adminpass" # Default password for the admin user
        with self.driver.session() as session:
            try:
                # Check if admin user already exists
                result = session.run("MATCH (u:User {username: $username}) RETURN u", username=admin_username)
                if result.single():
                    print(f"Admin user '{admin_username}' already exists in DB.")
                else:
                    # Create admin user
                    hashed_password = pbkdf2_sha256.hash(admin_password)
                    session.run(
                        "CREATE (u:User {username: $username, hashed_password: $hashed_password, approved: true}) RETURN u",
                        username=admin_username, hashed_password=hashed_password
                    )
                    print(f"Admin user '{admin_username}' created and approved with password '{admin_password}' (hashed).")
                # Ensure 'admin' role exists and link it to the admin user
                session.run("MERGE (r:Role {name: 'admin'}) RETURN r")
                link_result = session.run(
                    "MATCH (u:User {username: $username}) "
                    "MATCH (r:Role {name: 'admin'}) "
                    "MERGE (u)-[hr:HAS_ROLE]->(r) "
                    "RETURN hr",
                    username=admin_username
                )
                if link_result.single():
                    print(f"Admin user '{admin_username}' linked to 'admin' role.")
                else:
                    print(f"Admin user '{admin_username}' link to 'admin' role already exists or failed (no relationship created for MERGE).")
                return True
            except Exception as e:
                print(f"Error initializing admin user: {e}")
                return False
            
    def close(self):
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed.")

    def _create_node_if_not_exists(self, tx, node_id, label, node_type):
        query = f"""
        MERGE (n:{node_type} {{id: $node_id}})
        ON CREATE SET n.label = $label
        RETURN n
        """
        tx.run(query, node_id=node_id, label=label)

    def _create_relationship_if_not_exists(self, tx, source_id, target_id, rel_type, rel_label=""):
        query = f"""
        MATCH (source_node), (target_node)
        WHERE source_node.id = $source_id AND target_node.id = $target_id
        MERGE (source_node)-[r:{rel_type}]->(target_node)
        ON CREATE SET r.label = $rel_label
        RETURN r
        """
        tx.run(query, source_id=source_id, target_id=target_id, rel_type=rel_type, rel_label=rel_label)

    # --- User Management Methods ---
    def create_user(self, username: str, hashed_password: str) -> tuple[bool, str]:
        """Creates a new user node in Neo4j with an unapproved status."""
        if not self.driver:
            print("Neo4j driver not initialized. Cannot create user.")
            return False, "Database connection error."
        with self.driver.session() as session:
            try:
                result = session.run(
                    "MERGE (u:User {username: $username}) "
                    "ON CREATE SET u.hashed_password = $hashed_password, u.approved = false, u.created_at = datetime() "
                    "ON MATCH SET u.last_attempted_registration_at = datetime() " # Update timestamp if user already exists
                    "RETURN u",
                    username=username, hashed_password=hashed_password
                )
                if result.single():
                    print(f"User '{username}' registered (pending approval).")
                    return True, "Registration successful. Your account is pending administrator approval."
                return False, "Failed to register user (username might already exist or internal error)."
            except Exception as e:
                print(f"Error creating user in Neo4j: {e}")
                return False, f"Database error during registration: {e}"

    def get_user(self, username: str) -> dict | None:
        """Retrieves user data (username, hashed_password, approved) from Neo4j."""
        if not self.driver:
            return None
        with self.driver.session() as session:
            result = session.run(
                "MATCH (u:User {username: $username}) RETURN u.username AS username, u.hashed_password AS hashed_password, u.approved AS approved",
                username=username
            )
            record = result.single()
            return record.data() if record else None

    def update_user_approval(self, username: str, approved: bool) -> bool:
        """Updates the approval status of a user."""
        if not self.driver:
            return False
        with self.driver.session() as session:
            try:
                session.run(
                    "MATCH (u:User {username: $username}) SET u.approved = $approved RETURN u",
                    username=username, approved=approved
                )
                print(f"User '{username}' approval status set to {approved}.")
                return True
            except Exception as e:
                print(f"Error updating user approval status for '{username}': {e}")
                return False
    
    def get_unapproved_users(self) -> list[dict]:
        """Returns a list of users whose accounts are not yet approved."""
        if not self.driver:
            return []
        with self.driver.session() as session:
            result = session.run("MATCH (u:User) WHERE u.approved = false RETURN u.username AS username, u.created_at AS created_at ORDER BY u.created_at")
            return [record.data() for record in result]
    
    def check_user_role(self, username: str, role_name: str) -> bool:
        """Checks if a user has a specific role."""
        if not self.driver:
            return False
        with self.driver.session() as session:
            result = session.run(
                "MATCH (u:User {username: $username})-[:HAS_ROLE]->(r:Role {name: $role_name}) RETURN u",
                username=username, role_name=role_name
            )
            return bool(result.single())

    # --- Existing methods (kept for completeness) ---
    def store_code_generation_with_lineage(self, generation_id: str, original_code_snippet: str, generated_code_snippet: str, timestamp: str, flow_data: dict = None, metrics: dict = None) -> bool:
        """
        Stores code generation event and its associated data lineage (nodes and edges) in Neo4j.
        Optionally stores metrics.
        """
        if not self.driver:
            print("Neo4j driver not initialized. Cannot store data.")
            return False
        with self.driver.session() as session:
            try:
                # Create CodeGeneration node
                query_create_gen_node = """
                CREATE (g:CodeGeneration {
                    id: $generation_id,
                    original_code_snippet: $original_code,
                    generated_code_snippet: $generated_code,
                    timestamp: datetime($timestamp),
                    type: "CodeGeneration",
                    confidence: $confidence,
                    effort_hours: $effort_hours,
                    original_time_hours: $original_time_hours,
                    time_saved_hours: $time_saved_hours
                })
                RETURN g.id
                """
                session.write_transaction(lambda tx: tx.run(query_create_gen_node,
                                                        generation_id=generation_id,
                                                        original_code=original_code_snippet,
                                                        generated_code=generated_code_snippet,
                                                        timestamp=timestamp,
                                                        confidence=metrics.get("confidence") if metrics else None,
                                                        effort_hours=metrics.get("effort") if metrics else None,
                                                        original_time_hours=metrics.get("original_time") if metrics else None,
                                                        time_saved_hours=metrics.get("time_saved") if metrics else None
                                                        ).single()[0])
                # Store data lineage details and link to CodeGeneration event
                if flow_data and (flow_data.get("nodes") or flow_data.get("edges")):
                    def _process_lineage_tx(tx):
                        node_ids_involved = set()
                        # Create nodes
                        for node_data in flow_data.get("nodes", []):
                            node_id = node_data.get("id")
                            label = node_data.get("label", node_id)
                            node_type = node_data.get("type", "GenericNode") # Default node type
                            if node_id:
                                self._create_node_if_not_exists(tx, node_id, label, node_type)
                                node_ids_involved.add(node_id)
                        # Create relationships
                        for edge_data in flow_data.get("edges", []):
                            source_id = edge_data.get("source")
                            target_id = edge_data.get("target")
                            rel_type = edge_data.get("rel_type", "FLOWS_TO") # Default rel type
                            rel_label = edge_data.get("label", "")
                            if source_id and target_id:
                                self._create_relationship_if_not_exists(tx, source_id, target_id, rel_type, rel_label)
                                node_ids_involved.add(source_id)
                                node_ids_involved.add(target_id)
                        # Link the CodeGeneration event to all nodes involved in this lineage
                        query_link_gen_to_nodes = """
                        MATCH (g:CodeGeneration {id: $generation_id})
                        WITH g
                        UNWIND $node_ids AS node_id
                        MATCH (n) WHERE n.id = node_id
                        MERGE (g)-[:INVOLVED_IN_LINEAGE]->(n)
                        RETURN g
                        """
                        if node_ids_involved:
                            tx.run(query_link_gen_to_nodes, generation_id=generation_id, node_ids=list(node_ids_involved))
                    session.write_transaction(_process_lineage_tx)
                print(f"Code generation event {generation_id} and lineage stored successfully in Neo4j.")
                return True
            except Exception as e:
                print(f"Error storing code generation data in Neo4j: {e}")
                return False

    def store_chart_event(self, event_id: str, query: str, generated_code: str, data_preview: str, timestamp: str) -> bool:
        """
        Stores a chart generation event in Neo4j.
        """
        if not self.driver:
            print("Neo4j driver not initialized. Cannot store data.")
            return False
        with self.driver.session() as session:
            try:
                query_create_chart_node = """
                CREATE (c:ChartGeneration {
                    id: $event_id,
                    query: $user_query, # Changed parameter name to avoid conflict
                    generated_code: $generated_code,
                    data_preview: $data_preview,
                    timestamp: datetime($timestamp),
                    type: "ChartGeneration"
                })
                RETURN c.id
                """
                # FIX: Correctly pass the query string as the first argument, and then the parameters as keyword arguments.
                session.write_transaction(lambda tx: tx.run(query_create_chart_node,
                                                        event_id=event_id,
                                                        user_query=query, # Pass value to new parameter name
                                                        generated_code=generated_code,
                                                        data_preview=data_preview,
                                                        timestamp=timestamp).single()[0])
                print(f"Chart generation event {event_id} stored successfully in Neo4j.")
                return True
            except Exception as e:
                print(f"Error storing chart generation data in Neo4j: {e}")
                return False

    def store_document_query_event(self, event_id: str, query: str, answer: str, document_name: str, extracted_text_preview: str, timestamp: str) -> bool:
        """
        Stores a document query event in Neo4j.
        """
        if not self.driver:
            print("Neo4j driver not initialized. Cannot store data.")
            return False
        with self.driver.session() as session:
            try:
                query_create_doc_node = """
                CREATE (d:DocumentQuery {
                    id: $event_id,
                    query: $query,
                    answer: $answer,
                    document_name: $document_name,
                    extracted_text_preview: $extracted_text_preview,
                    timestamp: datetime($timestamp),
                    type: "DocumentQuery"
                })
                RETURN d.id
                """
                session.write_transaction(lambda tx: tx.run(query_create_doc_node,
                                                        event_id=event_id,
                                                        query=query,
                                                        answer=answer,
                                                        document_name=document_name,
                                                        extracted_text_preview=extracted_text_preview,
                                                        timestamp=timestamp).single()[0])
                print(f"Document query event {event_id} stored successfully in Neo4j.")
                return True
            except Exception as e:
                print(f"Error storing document query data in Neo4j: {e}")
                return False

    def store_project_flow_event(self, event_id: str, description: str, generated_code: str, flow_data: dict, timestamp: str, diagram_type: str = "Mermaid (Flowchart)") -> bool:
        """
        Stores project flow diagram generation event and its associated conceptual flow (nodes and edges) in Neo4j.
        Now accepts generated_code and diagram_type for more flexibility.
        """
        if not self.driver:
            print("Neo4j driver not initialized. Cannot store data.")
            return False
        with self.driver.session() as session:
            try:
                # Create ProjectFlowDiagram node
                query_create_flow_node = """
                CREATE (p:ProjectFlowDiagram {
                    id: $event_id,
                    description: $description,
                    generated_diagram_code: $generated_code,
                    diagram_type: $diagram_type,
                    timestamp: datetime($timestamp),
                    type: "ProjectFlowDiagram"
                })
                RETURN p.id
                """
                session.write_transaction(lambda tx: tx.run(query_create_flow_node,
                                                        event_id=event_id,
                                                        description=description,
                                                        generated_code=generated_code,
                                                        diagram_type=diagram_type,
                                                        timestamp=timestamp).single()[0])
                # Store data lineage details and link to ProjectFlowDiagram event
                # Only link nodes/edges if flow_data is meaningful (e.g., from Mermaid Flowchart analysis)
                if flow_data and (flow_data.get("nodes") or flow_data.get("edges")):
                    def _process_lineage_tx(tx):
                        node_ids_involved = set()
                        # Create nodes
                        for node_data in flow_data.get("nodes", []):
                            node_id = node_data.get("id")
                            label = node_data.get("label", node_id)
                            node_type = node_data.get("type", "GenericNode") # Default node type
                            if node_id:
                                self._create_node_if_not_exists(tx, node_id, label, node_type)
                                node_ids_involved.add(node_id)
                        # Create relationships
                        for edge_data in flow_data.get("edges", []):
                            source_id = edge_data.get("source")
                            target_id = edge_data.get("target")
                            rel_type = edge_data.get("rel_type", "FLOWS_TO") # Default rel type
                            rel_label = edge_data.get("label", "")
                            if source_id and target_id:
                                self._create_relationship_if_not_exists(tx, source_id, target_id, rel_type, rel_label)
                                node_ids_involved.add(source_id)
                                node_ids_involved.add(target_id)
                        # Link the ProjectFlowDiagram event to all nodes involved in this lineage
                        query_link_flow_to_nodes = """
                        MATCH (p:ProjectFlowDiagram {id: $event_id})
                        WITH p
                        UNWIND $node_ids AS node_id
                        MATCH (n) WHERE n.id = node_id
                        MERGE (p)-[:INVOLVED_IN_FLOW]->(n)
                        RETURN p
                        """
                        if node_ids_involved:
                            tx.run(query_link_flow_to_nodes, event_id=event_id, node_ids=list(node_ids_involved))
                    session.write_transaction(_process_lineage_tx)
                print(f"Project flow diagram event {event_id} and lineage stored successfully in Neo4j.")
                return True
            except Exception as e:
                print(f"Error storing project flow diagram data in Neo4j: {e}")
                return False

    def store_wireframe_event(self, event_id: str, description: str, generated_mukuro_code: str, timestamp: str) -> bool:
        """
        Stores a wireframe generation event in Neo4j.
        """
        if not self.driver:
            print("Neo4j driver not initialized. Cannot store data.")
            return False
        with self.driver.session() as session:
            try:
                query = """
                CREATE (w:WireframeGeneration {
                    id: $event_id,
                    description: $description,
                    generated_mukuro_code: $generated_mukuro_code,
                    timestamp: datetime($timestamp),
                    type: "WireframeGeneration"
                })
                RETURN w.id
                """
                session.write_transaction(lambda tx: tx.run(query,
                                                        event_id=event_id,
                                                        description=description,
                                                        generated_mukuro_code=generated_mukuro_code,
                                                        timestamp=timestamp).single()[0])
                print(f"Wireframe generation event {event_id} stored successfully in Neo4j.")
                return True
            except Exception as e:
                print(f"Error storing wireframe generation data in Neo4j: {e}")
                return False