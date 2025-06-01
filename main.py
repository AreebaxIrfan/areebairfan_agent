import os
import chainlit as cl
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, function_tool
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from typing import List, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

# Load environment variables
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Set up the OpenAI provider
provider = AsyncOpenAI(
    api_key=openai_api_key,
    base_url="https://api.openai.com/v1/",
)

# Define the model
model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",  # Use a compatible OpenAI model
    openai_client=provider,
)

# Run configuration
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

# Pydantic models for API response
class Contact(BaseModel):
    email: str
    linkedin: Optional[str] = None

class Project(BaseModel):
    id: int
    name: str
    description: str
    technologies: List[str]
    github_link: str

class Profile(BaseModel):
    name: str
    bio: str
    skills: List[str]
    contact: Contact
    projects: List[Project]
    hobbies: List[str]

# Tool to fetch profile data
@function_tool
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_profile() -> Profile:
    response = requests.get("https://areebaxirfan.vercel.app/api/profile")
    response.raise_for_status()
    return Profile(**response.json())

# Define the Areeba Irfan Agent
agent1 = Agent(
    name="AreebaIrfanAgent",
    instructions=(
        "You are the Areeba Irfan Agent, representing Areeba Irfan, an AI engineer. "
        "Use the fetch_profile tool to retrieve Areeba's data when needed. Answer questions only about: "
        "- Who Areeba can contact for professional networking or collaborations (use contact info from the profile). "
        "- How to arrange a meeting with Areeba (suggest emailing her contact.email). "
        "- Areeba’s free time activities (list hobbies from the profile). "
        "- Areeba’s career history, skills, or projects (use bio, skills, and projects from the profile). "
        "Respond in a friendly, professional tone. For irrelevant questions, politely say: "
        "'I'm sorry, I can only answer questions about Areeba Irfan’s professional connections, skills, projects, or hobbies.' "
        "Always ask a follow-up question to engage the user, e.g., 'Would you like to know more about Areeba’s projects?'"
    ),
    tools=[fetch_profile],
)

# Function to check question relevance (optional, as agent instructions handle this)
def is_relevant_question(question: str) -> bool:
    question = question.lower()
    relevant_keywords = [
        "contact", "meet", "meeting", "free time", "hobby", "hobbies", "past experience",
        "experience", "skill", "skills", "project", "projects", "career", "who", "how", "what"
    ]
    return any(keyword in question for keyword in relevant_keywords)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I'm the Areeba Irfan Agent. I can tell you about Areeba’s skills, projects, hobbies, or how to connect with her. How can I assist you?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    
    # Append user message to history
    history.append({"role": "user", "content": message.content})

    try:
        # Run the agent
        result = await Runner.run(
            agent1,
            input=history,
            run_config=run_config,
        )

        # Send the response
        response = str(result.final_output)
        await cl.Message(content=response).send()

        # Append assistant response to history
        history.append({"role": "assistant", "content": response})
        cl.user_session.set("history", history[-10:])  # Limit to last 10 messages

    except Exception as e:
        error_message = f"Sorry, an error occurred: {str(e)}. Please try again or ask a question about Areeba’s skills, projects, or hobbies."
        await cl.Message(content=error_message).send()
        history.append({"role": "assistant", "content": error_message})
        cl.user_session.set("history", history[-10:])

if __name__ == "__main__":
    cl.run()
