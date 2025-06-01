import os
import chainlit as cl
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

# Load environment variables
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Configure Gemini API
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

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

# Function to fetch profile data
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_profile() -> Profile:
    response = requests.get("https://areebaxirfan.vercel.app/api/profile")
    response.raise_for_status()
    return Profile(**response.json())

# Agent instructions
INSTRUCTIONS = (
    "You are the Areeba Irfan Agent, representing Areeba Irfan, an AI engineer. "
    "Use the provided profile data to answer questions only about: "
    "- Who Areeba can contact for professional networking or collaborations (use contact info from the profile). "
    "- How to arrange a meeting with Areeba (suggest emailing her contact.email). "
    "- Areeba’s free time activities (list hobbies from the profile). "
    "- Areeba’s career history, skills, or projects (use bio, skills, and projects from the profile). "
    "For irrelevant questions, politely say: "
    "'I'm sorry, I can only answer questions about Areeba Irfan’s professional connections, skills, projects, or hobbies.' "
    "Always ask a follow-up question to engage the user, e.g., 'Would you like to know more about Areeba’s projects?' "
    "Respond in a friendly, professional tone."
)

# Function to check if the question is relevant
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
    
    # Check if the question is relevant
    if not is_relevant_question(message.content):
        response = (
            "I'm sorry, I can only answer questions about who Areeba Irfan can contact, "
            "how to meet them, what they do in their free time, their past experience, skills, "
            "and projects. Please ask a relevant question."
        )
        await cl.Message(content=response).send()
        history.append({"role": "user", "content": message.content})
        history.append({"role": "assistant", "content": response})
        cl.user_session.set("history", history[-10:])  # Limit history to last 10 messages
        return

    # Append user message to history
    history.append({"role": "user", "content": message.content})

    try:
        # Fetch profile data
        profile = fetch_profile()

        # Prepare prompt with instructions, profile data, and history
        prompt = f"{INSTRUCTIONS}\n\nProfile Data:\n{profile.json()}\n\nConversation History:\n"
        for msg in history[-5:]:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        prompt += f"User: {message.content}\nAssistant:"

        # Call Gemini API
        response = model.generate_content(prompt)
        response_text = response.text

        # Send response
        await cl.Message(content=response_text).send()

        # Append assistant response to history
        history.append({"role": "assistant", "content": response_text})
        cl.user_session.set("history", history[-10:])

    except Exception as e:
        error_message = f"Sorry, an error occurred: {str(e)}. Please try again or ask a question about Areeba’s skills, projects, or hobbies."
        await cl.Message(content=error_message).send()
        history.append({"role": "assistant", "content": error_message})
        cl.user_session.set("history", history[-10:])

if __name__ == "__main__":
    cl.run()
