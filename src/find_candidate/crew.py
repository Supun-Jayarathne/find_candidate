import os
import yaml
import uuid
from crewai import Agent, Crew, Process, Task, Knowledge
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DirectoryReadTool, PDFSearchTool
from crewai.llm import LLM
from typing import List

@CrewBase
class FindCandidate:
    """FindCandidate crew for processing multiple CVs"""
    
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.agents_config = 'config/agents.yaml'
        self.tasks_config = 'config/tasks.yaml'
        
        # Initialize knowledge base with unique collection name
        collection_name = f"cv_database_{uuid.uuid4().hex[:8]}"
        self.cv_knowledge = Knowledge(
            collection_name=collection_name,
            sources=[],
            embeddings_options={
                "overwrite_existing": True  # Allow overwriting existing entries
            }
        )
        
        self.directory_tool = DirectoryReadTool()
        self.pdf_tool = PDFSearchTool()

    @agent
    def CVExtractionAgent(self) -> Agent:
        return Agent(
            config=self.agents_config["CVExtractionAgent"],
            tools=[self.directory_tool,self.pdf_tool],
            knowledge=self.cv_knowledge,
            verbose=True,
            llm=LLM(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
        )

    @agent
    def CVMatchingAgent(self) -> Agent:
        return Agent(
            config=self.agents_config["CVMatchingAgent"],
            knowledge=self.cv_knowledge,
            verbose=True,
            llm=LLM(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
        )

    def get_cv_files(self) -> List[dict]:
        """Get CV files information"""
        if not os.path.exists(self.folder_path):
            raise ValueError(f"CV folder not found: {self.folder_path}")
            
        cv_files = []
        for f in os.listdir(self.folder_path):
            if f.endswith(".pdf"):
                full_path = os.path.abspath(os.path.join(self.folder_path, f))
                print(f"Checking file: {full_path}")  # Debugging line
                if os.path.exists(full_path):
                    file_id = uuid.uuid4().hex[:8]
                    cv_files.append({
                        'id': file_id,
                        'filename': f,
                        'path': full_path
                    })
                    print(f"Found CV {file_id}: {full_path}")
                else:
                    print(f"Warning: File not accessible: {full_path}")
        
        if not cv_files:
            raise ValueError(f"No PDF files found in {self.folder_path}")
            
        return cv_files

    def create_extraction_tasks(self) -> List[Task]:
        """Create extraction tasks for all CVs"""
        tasks = []
        cv_files = self.get_cv_files()
        
        for cv_file in cv_files:
            print(f"Creating extraction task for CV: {cv_file['path']}")
            task = Task(
                config=self.tasks_config["ExtractCVDetails"],
                input_data={
                    "cv_id": cv_file['id'],
                    "cv_path": cv_file['path'],
                    "instructions": f"""
                    Extract information from CV: {cv_file['filename']}
                    
                    Use the following search queries with PDFSearchTool:
                    1. Name and Personal Info: {{"query": "name personal contact", "pdf": "{cv_file['path']}"}}
                    2. Education: {{"query": "education degree university", "pdf": "{cv_file['path']}"}}
                    3. Experience: {{"query": "experience work position", "pdf": "{cv_file['path']}"}}
                    4. Skills: {{"query": "skills technologies tools", "pdf": "{cv_file['path']}"}}
                    
                    Store the extracted information in Knowledge base with the CV ID: {cv_file['id']}
                    """
                },
                agent=self.CVExtractionAgent()
            )
            tasks.append(task)
        
        return tasks

    def create_matching_task(self, job_description: str) -> Task:
        """Create task for comparing CVs with job description"""
        return Task(
            config=self.tasks_config["CompareCVWithJobDescription"],
            input_data={
                "job_description": job_description,
                "instructions": "Analyze stored CV information and rank candidates based on job requirements."
            },
            agent=self.CVMatchingAgent()
        )

    def create_report_task(self) -> Task:
        """Create task for generating final report"""
        return Task(
            config=self.tasks_config["GenerateFinalReport"],
            input_data={
                "instructions": "Generate comprehensive report with rankings and recommendations."
            },
            output_file="report.md",
            agent=self.CVMatchingAgent()
        )

    @crew
    def crew(self, job_description: str) -> Crew:
        """Creates the FindCandidate crew"""
        try:
            extraction_tasks = self.create_extraction_tasks()
            compare_task = self.create_matching_task(job_description)
            report_task = self.create_report_task()
            
            all_tasks = extraction_tasks + [compare_task, report_task]
            
            return Crew(
                agents=[self.CVExtractionAgent(), self.CVMatchingAgent()],
                tasks=all_tasks,
                process=Process.sequential,
                verbose=True
            )
        except Exception as e:
            print(f"Error creating crew: {str(e)}")
            raise

    def cleanup_knowledge_base(self):
        """Clean up the knowledge base after processing"""
        try:
            self.cv_knowledge.clear()
            print("Knowledge base cleaned up successfully")
        except Exception as e:
            print(f"Error cleaning up knowledge base: {str(e)}")

    def run(self, job_description: str):
        """Main method to run the CV processing"""
        try:
            crew_instance = self.crew(job_description)
            result = crew_instance.run()
            self.cleanup_knowledge_base()  # Clean up after processing
            return result
        except Exception as e:
            print(f"Error during CV processing: {str(e)}")
            self.cleanup_knowledge_base()  # Clean up even if there's an error
            raise