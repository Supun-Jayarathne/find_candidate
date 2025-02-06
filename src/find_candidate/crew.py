import os
import yaml
from crewai import Agent, Crew, Process, Task, Knowledge
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import PDFSearchTool
from crewai.llm import LLM

@CrewBase
class FindCandidate:
    """FindCandidate crew"""

    # Path to folder containing CV PDFs
    cv_folder = "./CV"

    # Load agents and tasks configurations
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # Knowledge base to store extracted CV details
    cv_knowledge = Knowledge(
        collection_name="cv_database",
        sources=[]
    )

    # Initialize the PDF search tool
    pdf_tool = PDFSearchTool()

    # --- CV Extraction Agent ---
    @agent
    def CVExtractionAgent(self) -> Agent:
        return Agent(
            config=self.agents_config["CVExtractionAgent"],
            # role="CV Extraction and Summarization AI",
            # goal="Extract key details from CV PDFs and store structured summaries.",
            # backstory="An AI assistant for recruiters, extracting and organizing resume data.",
            # tools=[self.pdf_tool],
            knowledge=self.cv_knowledge,
            verbose=True,
            llm=LLM(model="gpt-4",api_key=os.environ.get("OPENAI_API_KEY"))
        )

    # --- CV Matching Agent ---
    @agent
    def CVMatchingAgent(self) -> Agent:
        return Agent(
            config=self.agents_config["CVMatchingAgent"],
            # role="CV Matching AI",
            # goal="Compare extracted CV details with the given job description to find the most suitable candidate.",
            # backstory="An AI hiring assistant that analyzes and ranks resumes based on job requirements.",
            knowledge=self.cv_knowledge,
            verbose=True,
            llm=LLM(model="gpt-4",api_key=os.environ.get("OPENAI_API_KEY"))
        )

    # --- CV Extraction Task ---
    @task
    def ExtractCVDetails(self) -> Task:
        """Extract and store CV details from PDFs."""
        for cv_file in os.listdir(self.cv_folder):
            print(f"Processing CV: {cv_file}")
            if cv_file.endswith(".pdf"):
                return Task(  # ❌ FIX: Returning only a single task
                    config=self.tasks_config["ExtractCVDetails"],
                    # description=f"Extract and summarize details from {cv_file}.",
                    # expected_output="Structured CV details including Name, Skills, Experience, and Education.",
                    input_data={"cv_file": os.path.join(self.cv_folder, cv_file)},
                    # agent=self.CVExtractionAgent(),
                )
    # @task
    # def ExtractCVDetails(self) -> list:
    #     """Extract and store CV details."""
    #     tasks = []
    #     for cv_file in os.listdir(self.cv_folder):
    #         print(f"Processing CV: {cv_file}")
    #         if cv_file.endswith(".pdf"):
    #             tasks.append(
    #                 Task(
    #                     # description=f"Extract and summarize details from {cv_file}.",
    #                     # expected_output="Structured CV details including Name, Skills, Experience, and Education.",
    #                     config=self.tasks_config["ExtractCVDetails"],
    #                     input_data={"cv_file": os.path.join(self.cv_folder, cv_file)},
    #                     # agent=self.CVExtractionAgent(),
    #                 )
    #             )
    #     return tasks  # ✅ Returns a list of `Task` objects

    @task
    def CompareCVWithJobDescription(self) -> Task:
        # print(f"Received job description: {job_description}")  # Debugging line to check the passed value
        return Task(
            config=self.tasks_config["CompareCVWithJobDescription"],
            # description="Compare extracted CVs with {job_description} and identify the most suitable candidate.",
            # expected_output="Ranked list of candidates with reasoning for selection.",
            # input_data={"job_description": job_description},
            agent=self.CVMatchingAgent(),
        )


    @task
    def GenerateFinalReport(self) -> Task:
        return Task(
            config=self.tasks_config["GenerateFinalReport"],
            # description="Generate a final report summarizing the best-matched CVs along with their strengths and weaknesses.",
            # expected_output="A structured summary of the most suitable candidates. Highlighted strengths, weaknesses, and suitability scores. A final recommendation of the best candidates.",
            output_file="report.md",
            agent=self.CVMatchingAgent(),
        )

    @crew
    def crew(self, job_description: str) -> Crew:
        """Creates the FindCandidate crew"""
        extraction_tasks = self.ExtractCVDetails()

        # Ensure the job_description is passed to CompareCVWithJobDescription when creating the task
        compare_cv_task = self.CompareCVWithJobDescription()

        return Crew(
            agents=[self.CVExtractionAgent(), self.CVMatchingAgent()],
            tasks=[extraction_tasks, compare_cv_task, self.GenerateFinalReport()],
            process=Process.sequential,
            verbose=True,
        )


    
    # @crew
    # def crew(self, job_description: str) -> Crew:
    #     """Creates the FindCandidate crew"""
    #     agents = [self.CVExtractionAgent, self.CVMatchingAgent]
    #     return Crew(
    #         agents=self.agents,
    #         tasks=self.tasks,
    #         # tasks=self.extraction_tasks() + [self.matching_task(job_description)],
    #         # tasks=self.ExtractCVDetails() + [self.CompareCVWithJobDescription(job_description), self.GenerateFinalReport()],
    #         process=Process.sequential,
    #         verbose=True,
    #     )
    # @crew
    # def crew(self, job_description: str) -> Crew:
    #     """Creates the FindCandidate crew"""
    #     agents = [self.CVExtractionAgent, self.CVMatchingAgent]
    #     extraction_tasks = self.ExtractCVDetails()
    #     matching_task = self.CompareCVWithJobDescription(job_description)
    #     final_report_task = self.GenerateFinalReport()
        
    #     return Crew(
    #         agents=agents,
    #         tasks=extraction_tasks + [matching_task, final_report_task],
    #         process=Process.sequential,
    #         verbose=True,
    #     )
