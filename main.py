import os
import chainlit as cl
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, function_tool
import requests
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

# Set up the provider for Gemini API

provider = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/",
)

# Define the model for Gemini
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",  # Ensure this is a valid Gemini model; verify with API docs
    openai_client=provider,
)

# Run configuration
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

@function_tool
def areeba_data():
    response = requests.get("https://areebaxirfan.vercel.app/api/profile")
    result = response.json()
    return result

# Main Areeba Irfan Agent
areeba_agent = Agent(
    name="areeba_agent",
    instructions="You are the areeba irfan assistant agent ,You should shared all the info about the areeba irfan related queries, you give all the answare related to the areeba irfan query , except there personal info by using the tool of areeba_data ,Your ans should be consice and tone should be humble and used easy english "
    "If user asked about areeba skill and project or experience you should to used the areeba_data to tool for giving the particular answare",
    tools=[areeba_data]
)

# Function to check if the question is relevant to Areeba Irfan
def is_relevant_question(question: str) -> bool:
    """
    Checks if the user’s question relates to Areeba Irfan’s profile, skills, projects, or contact info.
    
    Args:
        question: The user’s input question as a string.
    
    Returns:
        bool: True if the question is relevant, False otherwise.
    """
    question = question.lower().strip()
    relevant_keywords = [
        "areeba", "irfan", "contact", "meet", "meeting", "free time", "hobby", "hobbies", 
        "past experience", "experience", "skill", "skills", "project", "projects", "career", 
        "who", "how", "what", "hi", "hello", "data", "profile", "website", "portfolio", 
        "achievement", "hackathon", "coding", "background"
    ]
    return any(keyword in question for keyword in relevant_keywords)

@cl.on_chat_start
async def handle_chat_start():
    """
    Initializes the chat session and sends a welcome message.
    """
    cl.user_session.set("history", [])
    welcome_message = (
        "Hello! I’m the Areeba Irfan Agent. I can help with Areeba’s skills, projects, "
        "contact info, background, or profile data. How can I assist you today?"
    )
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def handle_message(message: cl.Message):
    """
    Handles incoming user messages, checks relevance, and processes responses using the Areeba Irfan Agent.
    
    Args:
        message: The incoming user message from Chainlit.
    """
    history = cl.user_session.get("history", [])
    
    # Check if the question is relevant
    if not is_relevant_question(message.content):
        response = (
            "I’m sorry, I can only answer questions about Areeba Irfan, such as her contact info, "
            "how to meet her, her free time or hobbies, past experience, skills, projects, or profile data. "
            "Please ask a relevant question!"
        )
        await cl.Message(content=response).send()
        history.append({"role": "user", "content": message.content})
        history.append({"role": "assistant", "content": response})
        cl.user_session.set("history", history[-10:])  # Keep last 10 messages
        return

    # Append user message to history
    history.append({"role": "user", "content": message.content})

    try:
        # Run the agent with the current history
        result = await Runner.run(
           areeba_agent,
            input=history,
            run_config=run_config,
        )

        # Send the agent’s response
        response = str(result.final_output)
        await cl.Message(content=response).send()

        # Append assistant response to history
        history.append({"role": "assistant", "content": response})
        cl.user_session.set("history", history[-10:])  # Keep last 10 messages

    except Exception as e:
        error_message = f"Sorry, I encountered an error while processing your request: {str(e)}"
        await cl.Message(content=error_message).send()
        history.append({"role": "user", "content": message.content})
        history.append({"role": "assistant", "content": error_message})
        cl.user_session.set("history", history[-10:])  # Keep last 10 messages
