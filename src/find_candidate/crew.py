import os
import yaml
import uuid
from crewai import Agent, Crew, Process, Task, Knowledge
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DirectoryReadTool, PDFSearchTool
from crewai.llm import LLM
from typing import List
# import openlit
from agentops import track_agent
import agentops

@CrewBase
class FindCandidate:
    """FindCandidate crew for processing multiple CVs"""
    
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.agents_config = 'config/agents.yaml'
        self.tasks_config = 'config/tasks.yaml'

        # openlit.init(disable_metrics=True)
        agentops.init()
        
        # Initialize unified knowledge base with unique collection name
        collection_name = f"candidate_matching_{uuid.uuid4().hex[:8]}"
        self.knowledge_base = Knowledge(
            collection_name=collection_name,
            sources=[],
            embeddings_options={
                "overwrite_existing": True,
                "model": "all-MiniLM-L6-v2"  # Efficient model for matching
            }
        )
        
        self.directory_tool = DirectoryReadTool()
        self.pdf_tool = PDFSearchTool()

    @track_agent(name='CVExtractionAgent')
    @agent
    def CVExtractionAgent(self) -> Agent:
        return Agent(
            config=self.agents_config["CVExtractionAgent"],
            tools=[self.directory_tool, self.pdf_tool],
            knowledge=self.knowledge_base,
            verbose=True,
            llm=LLM(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
        )

    @track_agent(name='CVMatchingAgent')
    @agent
    def CVMatchingAgent(self) -> Agent:
        return Agent(
            config=self.agents_config["CVMatchingAgent"],
            knowledge=self.knowledge_base,
            verbose=True,
            llm=LLM(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
        )
    
    @track_agent(name='SOPValidationAgent')
    @agent
    def SOPValidationAgent(self) -> Agent:
        """Agent for validating extracted CV details and matching results"""
        return Agent(
            config=self.agents_config["SOPValidationAgent"],
            knowledge=self.knowledge_base,  # Access knowledge base for validation
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
            task = Task(
                config=self.tasks_config["ExtractCVDetails"],
                input_data={
                    "cv_id": cv_file['id'],
                    "cv_path": cv_file['path'],
                    "instructions": f"""
                    Extract detailed information from CV: {cv_file['filename']}
                    
                    Use these structured search queries with PDFSearchTool:
                    1. Skills and Technologies: {{"query": "technical skills programming languages technologies frameworks tools", "pdf": "{cv_file['path']}"}}
                    2. Work Experience: {{"query": "work experience job history positions responsibilities achievements", "pdf": "{cv_file['path']}"}}
                    3. Education and Qualifications: {{"query": "education degrees certifications qualifications", "pdf": "{cv_file['path']}"}}
                    4. Projects and Achievements: {{"query": "projects achievements accomplishments", "pdf": "{cv_file['path']}"}}
                    
                    Store in Knowledge base with CV ID: {cv_file['id']}
                    Include metadata:
                    - document_type: "cv"
                    - file_name: "{cv_file['filename']}"
                    - extracted_skills: [list of identified skills]
                    - experience_years: [total years of experience]
                    - education_level: [highest education level]
                    """
                },
                agent=self.CVExtractionAgent()
            )
            tasks.append(task)
        
        return tasks

    def create_matching_task(self, job_description: str) -> Task:
        """Create task for comparing CVs with job description"""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        # Instead of add_entry, use the appropriate CrewAI Knowledge methods
        return Task(
            config=self.tasks_config["CompareCVWithJobDescription"],
            input_data={
                "job_description": job_description,
                "instructions": """
                Analyze the provided job description and compare with stored CV information.
                
                Perform detailed CV matching:
                1. Compare required skills with candidate skills
                2. Match experience levels
                3. Verify education requirements
                4. Analyze project relevance
                5. Consider overall fit
                
                Ranking criteria:
                - Skills match (40%)
                - Experience relevance (30%)
                - Education fit (20%)
                - Overall profile (10%)
                
                Use hybrid search to find best matches:
                1. Semantic search for context understanding
                2. Keyword matching for specific requirements
                3. Cross-reference experience levels
                
                Return ranked candidates with:
                - Match percentage
                - Key matching points
                - Any skill gaps
                - Specific strengths
                """
            },
            agent=self.CVMatchingAgent()
        )

    def create_report_task(self) -> Task:
        """Create task for generating final report"""
        return Task(
            config=self.tasks_config["GenerateFinalReport"],
            input_data={
                "instructions": """
                Generate detailed matching report:
                1. Top 3 candidates with match percentages
                2. Specific matching strengths for each
                3. Any skill gaps or areas of concern
                4. Recommendations for interviews
                5. Summary of why each candidate matches
                
                Format as markdown with sections:
                - Executive Summary
                - Candidate Rankings
                - Detailed Analysis
                - Recommendations
                """
            },
            output_file="matching_report.md",
            agent=self.CVMatchingAgent()
        )
    
    def create_validation_task(self) -> Task:
        """Create a task to validate extracted CVs and matching results based on SOPs"""
        return Task(
            config=self.tasks_config["SOPValidationTask"],
            input_data={
                "instructions": """
                Validate the extracted CV information and matching results against standard operating procedures (SOP).
                
                Key validation steps:
                1. Ensure all necessary fields are extracted (Skills, Experience, Education, etc.).
                2. Validate data consistency (e.g., work experience years should align).
                3. Verify the correctness of the matching process (e.g., do skill matches align with job requirements?).
                4. Identify any missing or incorrect data entries.
                
                If any issues are found, flag them for review.
                """
            },
            agent=self.SOPValidationAgent()
        )

    @crew
    def crew(self, job_description: str) -> Crew:
        """Creates the FindCandidate crew"""
        try:
            extraction_tasks = self.create_extraction_tasks()
            compare_task = self.create_matching_task(job_description)
            validation_task = self.create_validation_task()
            report_task = self.create_report_task()
            
            all_tasks = extraction_tasks + [compare_task, validation_task, report_task]
            
            return Crew(
                agents=[self.CVExtractionAgent(), self.CVMatchingAgent(), self.SOPValidationAgent()],
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
            self.knowledge_base.clear()
            print("Knowledge base cleaned up successfully")
        except Exception as e:
            print(f"Error cleaning up knowledge base: {str(e)}")

    def run(self, job_description: str):
        """Main method to run the CV processing"""
        try:
            crew_instance = self.crew(job_description)
            result = crew_instance.run()
            self.cleanup_knowledge_base()
            return result
        except Exception as e:
            print(f"Error during CV processing: {str(e)}")
            self.cleanup_knowledge_base()
            raise


# import os
# import yaml
# import uuid
# from crewai import Agent, Crew, Process, Task, Knowledge
# from crewai.project import CrewBase, agent, crew, task
# from crewai_tools import DirectoryReadTool, PDFSearchTool
# from crewai.llm import LLM
# from typing import List

# @CrewBase
# class FindCandidate:
#     """FindCandidate crew for processing multiple CVs"""
    
#     def __init__(self, folder_path):
#         self.folder_path = folder_path
#         self.agents_config = 'config/agents.yaml'
#         self.tasks_config = 'config/tasks.yaml'
        
#         # Initialize knowledge base with unique collection name
#         collection_name = f"cv_database_{uuid.uuid4().hex[:8]}"
#         self.cv_knowledge = Knowledge(
#             collection_name=collection_name,
#             sources=[],
#             embeddings_options={
#                 "overwrite_existing": True  # Allow overwriting existing entries
#             }
#         )
        
#         self.directory_tool = DirectoryReadTool()
#         self.pdf_tool = PDFSearchTool()

#     @agent
#     def CVExtractionAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config["CVExtractionAgent"],
#             tools=[self.directory_tool,self.pdf_tool],
#             knowledge=self.cv_knowledge,
#             verbose=True,
#             llm=LLM(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
#         )

#     @agent
#     def CVMatchingAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config["CVMatchingAgent"],
#             knowledge=self.cv_knowledge,
#             verbose=True,
#             llm=LLM(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
#         )

#     def get_cv_files(self) -> List[dict]:
#         """Get CV files information"""
#         if not os.path.exists(self.folder_path):
#             raise ValueError(f"CV folder not found: {self.folder_path}")
            
#         cv_files = []
#         for f in os.listdir(self.folder_path):
#             if f.endswith(".pdf"):
#                 full_path = os.path.abspath(os.path.join(self.folder_path, f))
#                 print(f"Checking file: {full_path}")  # Debugging line
#                 if os.path.exists(full_path):
#                     file_id = uuid.uuid4().hex[:8]
#                     cv_files.append({
#                         'id': file_id,
#                         'filename': f,
#                         'path': full_path
#                     })
#                     print(f"Found CV {file_id}: {full_path}")
#                 else:
#                     print(f"Warning: File not accessible: {full_path}")
        
#         if not cv_files:
#             raise ValueError(f"No PDF files found in {self.folder_path}")
            
#         return cv_files

#     def create_extraction_tasks(self) -> List[Task]:
#         """Create extraction tasks for all CVs"""
#         tasks = []
#         cv_files = self.get_cv_files()
        
#         for cv_file in cv_files:
#             print(f"Creating extraction task for CV: {cv_file['path']}")
#             task = Task(
#                 config=self.tasks_config["ExtractCVDetails"],
#                 input_data={
#                     "cv_id": cv_file['id'],
#                     "cv_path": cv_file['path'],
#                     "instructions": f"""
#                     Extract information from CV: {cv_file['filename']}
                    
#                     Use the following search queries with PDFSearchTool:
#                     1. Name and Personal Info: {{"query": "name personal contact", "pdf": "{cv_file['path']}"}}
#                     2. Education: {{"query": "education degree university", "pdf": "{cv_file['path']}"}}
#                     3. Experience: {{"query": "experience work position", "pdf": "{cv_file['path']}"}}
#                     4. Skills: {{"query": "skills technologies tools", "pdf": "{cv_file['path']}"}}
                    
#                     Store the extracted information in Knowledge base with the CV ID: {cv_file['id']}
#                     """
#                 },
#                 agent=self.CVExtractionAgent()
#             )
#             tasks.append(task)
        
#         return tasks

#     def create_matching_task(self, job_description: str) -> Task:
#         """Create task for comparing CVs with job description"""
#         # Save job description metadata in the knowledge base
#         self.cv_knowledge.add_entry(
#             entry_id="job_description",
#             content=job_description,
#             metadata={"type": "job_description"}
#         )
        
#         # Retrieve job description metadata from the knowledge base
#         job_description_metadata = self.cv_knowledge.get_entry("job_description")
        
#         return Task(
#             config=self.tasks_config["CompareCVWithJobDescription"],
#             input_data={
#                 "job_description": job_description_metadata["content"],
#                 "instructions": "Analyze stored CV information and rank candidates based on job requirements using hybrid search (keyword-based and semantic search)."
#             },
#             agent=self.CVMatchingAgent()
#         )

#     def create_report_task(self) -> Task:
#         """Create task for generating final report"""
#         return Task(
#             config=self.tasks_config["GenerateFinalReport"],
#             input_data={
#                 "instructions": "Generate comprehensive report with rankings and recommendations."
#             },
#             output_file="report.md",
#             agent=self.CVMatchingAgent()
#         )

#     @crew
#     def crew(self, job_description: str) -> Crew:
#         """Creates the FindCandidate crew"""
#         try:
#             extraction_tasks = self.create_extraction_tasks()
#             compare_task = self.create_matching_task(job_description)
#             report_task = self.create_report_task()
            
#             all_tasks = extraction_tasks + [compare_task, report_task]
            
#             return Crew(
#                 agents=[self.CVExtractionAgent(), self.CVMatchingAgent()],
#                 tasks=all_tasks,
#                 process=Process.sequential,
#                 verbose=True
#             )
#         except Exception as e:
#             print(f"Error creating crew: {str(e)}")
#             raise

#     def cleanup_knowledge_base(self):
#         """Clean up the knowledge base after processing"""
#         try:
#             self.cv_knowledge.clear()
#             print("Knowledge base cleaned up successfully")
#         except Exception as e:
#             print(f"Error cleaning up knowledge base: {str(e)}")

#     def run(self, job_description: str):
#         """Main method to run the CV processing"""
#         try:
#             crew_instance = self.crew(job_description)
#             result = crew_instance.run()
#             self.cleanup_knowledge_base()  # Clean up after processing
#             return result
#         except Exception as e:
#             print(f"Error during CV processing: {str(e)}")
#             self.cleanup_knowledge_base()  # Clean up even if there's an error
#             raise