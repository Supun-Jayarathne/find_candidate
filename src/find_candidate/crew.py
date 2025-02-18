import os
import yaml
import uuid
from crewai import Agent, Crew, Process, Task, Knowledge
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DirectoryReadTool, PDFSearchTool
from crewai.crews.crew_output import CrewOutput
from langchain_docling import DoclingLoader
from crewai.llm import LLM
from typing import List
# import openlit
# from agentops import track_agent
# import agentops

@CrewBase
class FindCandidate:
    """FindCandidate crew for processing multiple CVs"""
    
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.agents_config = 'config/agents.yaml'
        self.tasks_config = 'config/tasks.yaml'

        # openlit.init(disable_metrics=True)
        # agentops.init(model="gpt-4o-mini")
        
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
        
        self.directory_tool = DirectoryReadTool(directory=self.folder_path)
        self.pdf_tool = PDFSearchTool()
        # self.docling_tool = DoclingLoader()

    # @track_agent(name='CVFilteringAgent')
    @agent
    def CVFilteringAgent(self) -> Agent:
        """Agent responsible for filtering CVs based on expected job title"""
        return Agent(
            config=self.agents_config["CVFilteringAgent"],
            tools=[self.directory_tool, self.pdf_tool],
            knowledge=self.knowledge_base,
            verbose=True,
            instructions="""
            You are responsible for filtering CVs based on expected job titles.
            
            IMPORTANT: When using the Search a PDF's content tool (PDFSearchTool):
            1. The 'query' parameter must be an actual search string, like "job title" or "position sought"
            2. The 'pdf' parameter must be the file path to a PDF file, e.g., "/path/to/resume.pdf"
            
            Example of CORRECT usage:
            pdf_tool.run(query="job title", pdf="/path/to/resume.pdf")
            
            Example of INCORRECT usage:
            pdf_tool.run(query={"description": "job title"}, pdf={"description": "/path/to/resume.pdf"})
            
            Always pass simple string values, never dictionaries or complex objects.
            """,
            llm=LLM(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
        )

    # @track_agent(name='CVExtractionAgent')
    @agent
    def CVExtractionAgent(self) -> Agent:
        return Agent(
            config=self.agents_config["CVExtractionAgent"],
            # tools=[self.directory_tool, self.pdf_tool],
            # tools=[self.directory_tool, self.docling_tool],
            tools=[self.directory_tool],
            knowledge=self.knowledge_base,
            verbose=True,
            llm=LLM(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
        )
    
    # @track_agent(name='CVEvaluationAgent')
    @agent
    def CVEvaluationAgent(self) -> Agent:
        """Agent for evaluating skill proficiency from CVs"""
        return Agent(
            config=self.agents_config["CVEvaluationAgent"],
            knowledge=self.knowledge_base,
            verbose=True,
            llm=LLM(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
        )


    # @track_agent(name='CVMatchingAgent')
    @agent
    def CVMatchingAgent(self) -> Agent:
        return Agent(
            config=self.agents_config["CVMatchingAgent"],
            knowledge=self.knowledge_base,
            verbose=True,
            llm=LLM(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
        )
    
    # @track_agent(name='SOPValidationAgent')
    @agent
    def SOPValidationAgent(self) -> Agent:
        """Agent for validating extracted CV details and matching results"""
        return Agent(
            config=self.agents_config["SOPValidationAgent"],
            knowledge=self.knowledge_base,  # Access knowledge base for validation
            verbose=True,
            llm=LLM(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
        )
    
    # @track_agent(name='RankingAgent')
    @agent
    def RankingAgent(self) -> Agent:
        """Agent responsible for ranking candidates"""
        return Agent(
            config=self.agents_config["RankingAgent"],
            knowledge=self.knowledge_base,
            verbose=True,
            llm=LLM(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
        )
    
    # @track_agent(name='ReportGenerationAgent')
    @agent
    def ReportGenerationAgent(self) -> Agent:
        """Agent responsible for generating the final candidate report"""
        return Agent(
            config=self.agents_config["ReportGenerationAgent"],
            knowledge=self.knowledge_base,
            verbose=True,
            llm=LLM(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"), temperature=0)
        )

    def create_cv_filtering_task(self, job_description: str, folder_path: str) -> Task:
        """Task to filter CVs based on expected job titles before extraction"""
        return Task(
            config=self.tasks_config["FilterCVsByJobTitle"],
            input_data={
                "job_description": job_description,
                "folder_path": folder_path,
                "instructions": f"""
                Scan all CVs in the folder `{folder_path}` and filter candidates based on their expected job title.

                Steps:
                1. Read each CV file in `{folder_path}`.
                2. For each CV file, use the PDFSearchTool with a specific query like:
                "job title", "position", "role".
                3. Compare the extracted job titles with the provided job description:
                "{job_description}"
                4. Select only CVs that match the job description.
                5. Output a structured list of valid CVs.

                When using the PDFSearchTool, make sure to provide two separate parameters:
                - query: A string containing your search query
                - pdf: A string containing the path to the PDF file

                Output Format:
                - Candidate Name
                - Expected Job Title
                - CV File Path
                - Filtering Status: Accepted / Rejected
                """
            },
            agent=self.CVFilteringAgent()
        )
    
    def create_extraction_tasks(self, filtered_cvs: List) -> List[Task]:
        """Create extraction tasks for only filtered CVs"""
        tasks = []
        
        if not filtered_cvs:
            print("Warning: No filtered CVs provided to create_extraction_tasks")
            return tasks

        # Handle various input formats
        if isinstance(filtered_cvs, str):
            print(f"Converting string to list: {filtered_cvs}")
            filtered_cvs = self.parse_filtering_results(filtered_cvs)
        
        # Additional debug info
        print(f"Processing {len(filtered_cvs)} filtered CVs")
        print(f"First item type: {type(filtered_cvs[0]) if filtered_cvs else 'N/A'}")

        for idx, cv in enumerate(filtered_cvs):
            try:
                # Handle various item formats
                if isinstance(cv, list) and len(cv) == 1:
                    cv = cv[0]  # Unpack list with single item
                
                # Extract ID and path
                if isinstance(cv, tuple) and len(cv) >= 2:
                    cv_id, cv_path = cv[0], cv[1]
                elif isinstance(cv, dict):
                    cv_id = cv.get('id', f'cv_{idx}')
                    cv_path = cv.get('path')
                else:
                    print(f"Skipping invalid CV format at index {idx}: {cv}")
                    continue
                
                # Validate ID and path
                if not cv_id:
                    cv_id = f'cv_{idx}'
                if not cv_path or not isinstance(cv_path, str):
                    print(f"Warning: Invalid file path for CV ID {cv_id}. Skipping...")
                    continue
                
                # Create task
                filename = os.path.basename(cv_path)
                
                # Extract text safely
                try:
                    docling_loader = DoclingLoader(file_path=cv_path)
                    extracted_text = docling_loader.load()
                except Exception as e:
                    print(f"Error extracting text from {filename}: {e}")
                    print(f"Continuing without extracted text for {cv_id}")
                    extracted_text = None
                
                task = Task(
                    config=self.tasks_config["ExtractCVDetails"],
                    input_data={
                        "cv_id": cv_id,
                        "cv_path": cv_path,
                        "extracted_text": extracted_text,
                        "instructions": f"""
                        Extract detailed information from the selected CV: {filename}
                        
                        {f"Use the provided extracted text" if extracted_text else "Read the CV file directly"}
                        to analyze:
                        - Skills and Technologies
                        - Work Experience
                        - Education and Certifications
                        - Projects and Achievements
                        
                        Store extracted data in the Knowledge Base with CV ID: {cv_id}
                        """
                    },
                    agent=self.CVExtractionAgent()
                )
                tasks.append(task)
                
            except Exception as e:
                print(f"Error creating task for CV at index {idx}: {str(e)}")
        
        print(f"Successfully created {len(tasks)} extraction tasks")
        return tasks


    # def create_extraction_tasks(self, filtered_cvs: List) -> List[Task]:
    #     """Create extraction tasks for only filtered CVs"""
    #     tasks = []

    #     for cv in filtered_cvs:
    #         # Ensure `cv` is in the correct format
    #         if isinstance(cv, list) and len(cv) == 2:  
    #             cv_id, cv_path = cv  # Unpack if it's a list of two elements
    #         elif isinstance(cv, tuple) and len(cv) == 2:
    #             cv_id, cv_path = cv  # Unpack if it's a tuple
    #         elif isinstance(cv, dict):
    #             cv_id = cv.get('id', 'unknown_id')
    #             cv_path = cv.get('path')
    #         else:
    #             print(f"Skipping invalid CV format: {cv}")
    #             continue

    #         # Ensure `cv_path` is a string, not a list
    #         if isinstance(cv_path, list):
    #             cv_path = cv_path[0] if cv_path else None  # Take the first element if it's a list

    #         # Check if cv_path is valid
    #         if not isinstance(cv_path, str) or not cv_path:
    #             print(f"Warning: Invalid file path for CV ID {cv_id}. Skipping...")
    #             continue

    #         filename = os.path.basename(cv_path)

    #         # Now use correct variables
    #         task = Task(
    #             config=self.tasks_config["ExtractCVDetails"],
    #             input_data={
    #                 "cv_id": cv_id,
    #                 "cv_path": cv_path,
    #                 "instructions": f"""
    #                 Extract detailed information from the selected CV: {filename}

    #                 Use **PDFSearchTool** to parse and extract:
    #                 - Skills and Technologies
    #                 - Work Experience
    #                 - Education and Certifications
    #                 - Projects and Achievements
                    
    #                 Store extracted data in the Knowledge Base with CV ID: {cv_id}
    #                 """
    #             },
    #             agent=self.CVExtractionAgent()
    #         )
    #         tasks.append(task)

    #     print(f"Created {len(tasks)} extraction tasks for filtered CVs")
    #     return tasks



    
    def create_evaluation_task(self) -> Task:
        """Create a task to evaluate candidates' skills based on CV information"""
        return Task(
            config=self.tasks_config["EvaluateSkillProficiency"],
            input_data={
                "instructions": """
                Evaluate the technical skills of candidates based on extracted CV data.

                - Cross-check listed skills with experience and projects.
                - Assign a proficiency level: Beginner, Intermediate, or Expert.
                - Justify each assessment with relevant evidence from CV data.
                """
            },
            agent=self.CVEvaluationAgent()
        )


    def create_matching_task(self, job_description: str) -> Task:
        """Create task for comparing CVs with job description"""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        # print("Current Knowledge Base Entries:")
        # print(self.knowledge_base.list_entries())  # Check stored data

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
    
    def create_validation_task(self) -> Task:
        """Create a task to validate extracted CVs based on SOPs"""
        return Task(
            config=self.tasks_config["SOPValidationTask"],
            input_data={
                "instructions": """
                Validate the extracted CV data and flag missing sections.

                Steps:
                1. Check for missing fields: Skills, Experience, Education, Certifications.
                2. If sections are missing, **add a flag** but continue processing.
                3. Assign a **Validation Status**:
                - **Pass**: If all key sections are present.
                - **Needs Review**: If some details are missing but the CV can still be ranked.
                4. Include a structured output:
                - **Validation Status**
                - **Missing Fields (if any)**
                - **Recommendation: Adjust ranking score if needed**.
                """
            },
            agent=self.SOPValidationAgent()
        )
    
    def create_ranking_task(self) -> Task:
        """Create a task to rank candidates based on evaluation results"""
        return Task(
            config=self.tasks_config["RankCandidatesTask"],
            input_data={
                "instructions": """
                Rank candidates using available data, even if some details are missing.

                1. Use scores from **CVMatchingAgent** and **SOPValidationAgent**.
                2. If key details are missing, adjust ranking score **but do not exclude the CV**.
                3. Assign rankings based on:
                - **Skills match (40%)**
                - **Experience relevance (30%)**
                - **Education fit (20%)**
                - **Overall profile completeness (10%)**
                4. Include a flag for missing details:
                - **Candidate Name**
                - **Overall Match Percentage**
                - **Missing Fields (if any)**
                - **Final Rank**
                """
            },
            agent=self.RankingAgent()
        )

    def create_final_report_task(self) -> Task:
        """Create a task to generate the final candidate report"""
        return Task(
            config=self.tasks_config["GenerateFinalReportTask"],
            input_data={
                "instructions": """
                Generate a structured report summarizing the top 5 ranked candidates.

                Include:
                - **Executive Summary**
                - **Top 5 Ranked Candidates**
                - **Detailed Candidate Analysis**
                - **Missing Information Section**
                - **Final Hiring Recommendations**

                Format the report as markdown and save as `final_candidates_report.md`.
                """
            },
            output_file="final_candidates_report.md",
            agent=self.ReportGenerationAgent()
        )
    
    def parse_filtering_results(self, results_text: str) -> List[tuple]:
        """Parses text output from filtering agent and converts it to structured data."""
        valid_cvs = []
        
        # Look for sections in the text that might be formatted in different ways
        # 1. Check for list of CVs in standard format
        for line in results_text.split("\n"):
            if "Accepted" in line and "|" in line:  
                try:
                    parts = line.split("|")
                    cv_id = parts[0].strip()
                    cv_path = parts[1].strip()
                    if cv_id and cv_path:
                        valid_cvs.append((cv_id, cv_path))
                except IndexError:
                    continue
        
        # 2. If no results found, try other format patterns
        if not valid_cvs:
            # Look for patterns like "ID: cv_123, Path: /path/to/file.pdf"
            import re
            pattern = r'ID:?\s*([^\s,]+).*?Path:?\s*([^\s,]+)'
            matches = re.findall(pattern, results_text)
            valid_cvs.extend(matches)
        
        return valid_cvs

    @crew
    def crew(self, job_description: str) -> Crew:
        """Creates the FindCandidate crew"""
        try:
            # Step 1: Run filtering task
            filtering_task = self.create_cv_filtering_task(job_description, self.folder_path)

            # Step 2: Initialize and run filtering crew
            filtering_crew = Crew(
                agents=[self.CVFilteringAgent()],
                tasks=[filtering_task],
                process=Process.sequential,
                verbose=True
            )

            filtering_results = filtering_crew.kickoff(inputs={'job_description': job_description})

            # Extract the actual response from CrewOutput
            filtered_cvs = []
            if isinstance(filtering_results, CrewOutput):
                print("Processing CrewOutput...")
                
                # Get string representation
                filtering_results_str = str(filtering_results)
                
                # Try to extract parsed results directly
                if hasattr(filtering_results, 'result') and filtering_results.result:
                    if isinstance(filtering_results.result, list):
                        filtered_cvs = filtering_results.result
                    else:
                        filtered_cvs = self.parse_filtering_results(str(filtering_results.result))
                elif hasattr(filtering_results, 'output') and filtering_results.output:
                    if isinstance(filtering_results.output, list):
                        filtered_cvs = filtering_results.output
                    else:
                        filtered_cvs = self.parse_filtering_results(str(filtering_results.output))
                else:
                    # Fallback to parsing string representation
                    filtered_cvs = self.parse_filtering_results(filtering_results_str)
            else:
                # If it's already a string or list
                if isinstance(filtering_results, list):
                    filtered_cvs = filtering_results
                else:
                    filtered_cvs = self.parse_filtering_results(str(filtering_results))
            
            # Safety check - ensure we have a list
            if not isinstance(filtered_cvs, list):
                print(f"Warning: filtered_cvs is not a list. Got {type(filtered_cvs)}. Converting...")
                if filtered_cvs:
                    filtered_cvs = [filtered_cvs]
                else:
                    filtered_cvs = []
            
            # Print for debugging
            print(f"Found {len(filtered_cvs)} filtered CVs to process")
            
            # Step 3: Create extraction tasks only for valid CVs
            extraction_tasks = self.create_extraction_tasks(filtered_cvs)

            # Step 4: Other processing tasks
            compare_task = self.create_matching_task(job_description)
            evaluation_task = self.create_evaluation_task()
            validation_task = self.create_validation_task()
            ranking_task = self.create_ranking_task()
            report_task = self.create_final_report_task()

            all_tasks = extraction_tasks + [compare_task, evaluation_task, validation_task, ranking_task, report_task]

            final_crew = Crew(
                agents=[
                    self.CVExtractionAgent(),
                    self.CVMatchingAgent(),
                    self.SOPValidationAgent(),
                    self.CVEvaluationAgent(),
                    self.RankingAgent(),
                    self.ReportGenerationAgent()
                ],
                tasks=all_tasks,
                process=Process.sequential,
                verbose=True
            )

            return final_crew

        except Exception as e:
            print(f"Error creating crew: {str(e)}")
            raise






    # @crew
    # def crew(self, job_description: str) -> Crew:
    #     """Creates the FindCandidate crew"""
    #     try:
    #         # Step 1: Filtering task
    #         filtering_task = self.create_cv_filtering_task(job_description, self.folder_path)

    #         # Step 2: Extraction tasks (initialized but will be populated after filtering)
    #         extraction_tasks = []

    #         # Step 3: Other processing tasks
    #         compare_task = self.create_matching_task(job_description)
    #         evaluation_task = self.create_evaluation_task()
    #         validation_task = self.create_validation_task()
    #         ranking_task = self.create_ranking_task()
    #         report_task = self.create_final_report_task()

    #         all_tasks = [filtering_task] + extraction_tasks + [compare_task, evaluation_task, validation_task, ranking_task, report_task]

    #         crew_instance = Crew(
    #             agents=[
    #                 self.CVFilteringAgent(),
    #                 self.CVExtractionAgent(),
    #                 self.CVMatchingAgent(),
    #                 self.SOPValidationAgent(),
    #                 self.CVEvaluationAgent(),
    #                 self.RankingAgent(),
    #                 self.ReportGenerationAgent()
    #             ],
    #             tasks=all_tasks,
    #             process=Process.sequential,
    #             verbose=True
    #         )

    #         # Step 4: Execute the Crew and get results
    #         crew_results = crew_instance.kickoff(inputs={'job_description': job_description})

    #         # Step 5: Extract filtered CVs from results (assuming filtering_task provides it)
    #         filtered_cvs = crew_results  # CrewOutput stores results in .outputs

    #         if not filtered_cvs:
    #             print("No relevant CVs found. Stopping process.")
    #             return None  # Stop execution if no CVs match the job description

    #         # Step 6: Create extraction tasks only for filtered CVs
    #         extraction_tasks = self.create_extraction_tasks(filtered_cvs)

    #         # Step 7: Re-run the crew with extraction tasks included
    #         final_tasks = extraction_tasks + [compare_task, evaluation_task, validation_task, ranking_task, report_task]
    #         final_crew = Crew(
    #             agents=[
    #                 self.CVExtractionAgent(),
    #                 self.CVMatchingAgent(),
    #                 self.SOPValidationAgent(),
    #                 self.CVEvaluationAgent(),
    #                 self.RankingAgent(),
    #                 self.ReportGenerationAgent()
    #             ],
    #             tasks=final_tasks,
    #             process=Process.sequential,
    #             verbose=True
    #         )

    #         return final_crew

    #     except Exception as e:
    #         print(f"Error creating crew: {str(e)}")
    #         raise



    # @crew
    # def crew(self, job_description: str) -> Crew:
    #     """Creates the FindCandidate crew"""
    #     try:
    #         extraction_tasks = self.create_extraction_tasks()
    #         compare_task = self.create_matching_task(job_description)
    #         evaluation_task = self.create_evaluation_task()
    #         validation_task = self.create_validation_task()
    #         ranking_task = self.create_ranking_task()
    #         report_task = self.create_final_report_task()
            
    #         all_tasks = extraction_tasks + [compare_task, evaluation_task, validation_task, ranking_task, report_task]
    #         # all_tasks = extraction_tasks + [compare_task, evaluation_task, ranking_task, report_task]
    #         # all_tasks = extraction_tasks + [evaluation_task, compare_task, validation_task, report_task]
    #         # all_tasks = extraction_tasks + [compare_task, report_task]
            
    #         return Crew(
    #         agents=[
    #             self.CVExtractionAgent(),
    #             self.CVMatchingAgent(),
    #             self.SOPValidationAgent(),
    #             self.CVEvaluationAgent(),
    #             self.RankingAgent(),
    #             self.ReportGenerationAgent()
    #         ],
    #             # agents=[self.CVExtractionAgent(), self.CVMatchingAgent()],
    #             tasks=all_tasks,
    #             process=Process.sequential,
    #             verbose=True
    #         )
    #     except Exception as e:
    #         print(f"Error creating crew: {str(e)}")
    #         raise

    # def cleanup_knowledge_base(self):
    #     """Clean up the knowledge base after processing"""
    #     try:
    #         self.knowledge_base.clear()
    #         print("Knowledge base cleaned up successfully")
    #     except Exception as e:
    #         print(f"Error cleaning up knowledge base: {str(e)}")

    # def run(self, job_description: str):
    #     """Main method to run the CV processing"""
    #     try:
    #         crew_instance = self.crew(job_description)
    #         result = crew_instance.run()
    #         self.cleanup_knowledge_base()
    #         return result
    #     except Exception as e:
    #         print(f"Error during CV processing: {str(e)}")
    #         self.cleanup_knowledge_base()
    #         raise


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