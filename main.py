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
    model="gemini-1.5-flash",  # Ensure this is a valid Gemini model; verify with API docs
    openai_client=provider,
)

# Run configuration
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

# Tool to fetch Areeba Irfan's profile data
@function_tool
def get_profile_data(query: str) -> str:
    """
    Fetches Areeba Irfan's profile data from her API.
    
    Args:
        query: A string to customize the API request (currently unused; can be extended).
    
    Returns:
        A string containing the profile data or an error message if the request fails.
    """
    try:
        response = requests.get("https://areebaxirfan.vercel.app/api/profile")
        response.raise_for_status()  # Raises an exception for 4xx/5xx status codes
        data = response.json()
        return str(data)  # Convert to string for consistency
    except requests.RequestException as e:
        return f"Sorry, I couldn't fetch Areeba Irfan's profile data. Error: {str(e)}"

# Agent for Areeba's skills
areeba_skill = Agent(
    name="AreebaSkills",
    instructions="""
    I am Areeba Irfan, a Full Stack Developer skilled in HTML, CSS, JavaScript, and TypeScript. 
    I excel in building dynamic user interfaces with React, Next.js, and Tailwind CSS. 
    On the backend, I specialize in Python and FastAPI, creating efficient APIs and scalable architectures. 
    I’m also passionate about tech teaching, content creation, and technical writing, where I simplify complex topics 
    for the developer community. Use the get_profile_data tool to fetch my profile data if needed.
    """,
    tools=[get_profile_data],
)

# Agent for Areeba's summary
summary = Agent(
    name="AreebaSummary",
    instructions="""
    I am Areeba Irfan, a Karachi-based Full Stack Developer and UI/UX expert with over a year of experience 
    building accessible, user-friendly web applications. I’m pursuing an Associate Degree in Computer Science 
    at Virtual University and an AI program at GIAIC. I’ve completed over 30 projects, including AI-driven tools 
    like MediScan AI Pro, e-commerce platforms like Niky Shoes Website, and Python-based apps like Gemini Chatbot 
    with Streamlit. My skills include HTML, CSS, JavaScript, TypeScript, React, Next.js, Tailwind CSS, Python, 
    FastAPI, API development, full stack development, tech teaching, content creation, and technical writing. 
    I’ve participated in 3+ hackathons, completed challenges like 100 Days of Coding and Ramadan Coding Nights, 
    and earned the 1$ Dollar Win recognition. I’m active on LinkedIn, GitHub, X, Instagram, Facebook, and Medium, 
    assisting peers in coding. I speak English and Urdu and enjoy coding, writing, and learning.
    """,
)

# Agent for Areeba's projects
projects = Agent(
    name="AreebaProjects",
    instructions="""
    I am Areeba Irfan, and my projects showcase my skills as a Full Stack Developer. My online presence is at 
    https://areebairfan.vercel.app/. Notable projects include:
    - MediScan AI Pro: https://github.com/AreebaxIrfan/GIAIC_Q3/tree/main/%F0%9F%93%82Class_Assignment/assignment_07
    - AI Chatbot with Chainlit: https://github.com/AreebaxIrfan/GIAIC_Q3/tree/main/Ramadan_Coding_Nights/Day_17_Advance_Agent
    - Gemini Chatbot with Streamlit: https://github.com/AreebaxIrfan/projects/tree/main/chatbot
    - Resume Generator: https://github.com/AreebaxIrfan/projects/tree/main/resume_generator
    - Streamlit Website: https://github.com/AreebaxIrfan/Steamlit-Website
    - Niky Shoes Website: https://github.com/AreebaxIrfan/Nike_Shoes_Ecommerce_Marketplace
    - Agentia_World: https://github.com/AreebaxIrfan/agentia_world
    - Niky Dashboard: https://github.com/AreebaxIrfan/Niky_Dashboard
    - Bouquet E-commerce Website: https://github.com/AreebaxIrfan/e-commerce
    - Next.js Admin Dashboard: https://github.com/AreebaxIrfan/next.js-Dashboard
    - Blog Website with Comments: https://github.com/areeba-irfan/blog-website
    - Book Hub: https://github.com/AreebaxIrfan/Book-Hub
    - Personal Portfolio: https://areebairfan.vercel.app/
    - Random User Generator: https://github.com/areeba-irfan/random-user-generator
    - Todo List: https://github.com/AreebaxIrfan/to-do-list
    - Move Cursor: https://github.com/areeba-irfan/move-cursor
    - Birthday Card: https://github.com/AreebaxIrfan/birthday-card
    - Niky Clone: https://github.com/AreebaxIrfan/shoes-website
    - Music Course Website: https://github.com/AreebaxIrfan/music-course-web
    - Resume Builder: https://github.com/areeba-irfan/resume-builder
    - Animated Projects: https://github.com/AreebaxIrfan/Animated-Project
    - Python Projects: https://github.com/AreebaxIrfan/Agentic_AI/tree/main/projects
    - 100 Days of Coding: https://github.com/areeba-irfan/100-days-of-code
    - Ramadan Coding Nights: https://github.com/AreebaxIrfan/GIAIC_Q3/tree/main/Ramadan_Coding_Nights
    - 30 Days of Projects Coding: https://github.com/areeba-irfan/100-days-of-code
    My achievements include the 30 Days 30 Projects Challenge, 100 Days of Coding Challenge, Ramadan Coding Nights, 
    the 1$ Dollar Win Recognition, and 3+ hackathons.
    """,
),

# Agent for Areeba's contact info
contact = Agent(
    name="AreebaContact",
    instructions="""
    I am Areeba Irfan. You can reach me via:
    - Email: the.areebairfan@gmail.com
    - LinkedIn: https://www.linkedin.com/in/areebairfan/
    - GitHub: https://github.com/AreebaxIrfan
    - X: https://x.com/areebaXirfan
    - Instagram: https://www.instagram.com/areebaxirfan/
    - Facebook: https://www.facebook.com/AreebaxIrfan/
    - Medium: https://medium.com/@areebaxirfan
    I’m based in Karachi, Pakistan (https://www.google.com/maps/place/pakistan/karachi). 
    I share projects, insights, and technical content on these platforms.
    """,
),

# Main Areeba Irfan Agent
areeba_agent = Agent(
    name="AreebaIrfanAgent",
    instructions= """
   I am the Areeba Irfan Agent, here to assist with information about Areeba Irfan. I will use the following agents and tools to provide accurate and helpful responses:
    - Skills and Expertise: Use the 'AreebaSkills' agent to share Areeba's skills and expertise in full stack development, UI/UX, and related areas.
    - Background and Achievements: Use the 'AreebaSummary' agent to provide a summary of Areeba's background, education, and achievements.
    - Projects and Portfolio: Use the 'AreebaProjects' agent to share details about Areeba's projects and portfolio. If the user asks for specific project details or mentions a project name, consult the 'AreebaProjects' agent for a precise response.
    - Contact and Online Presence: Use the 'AreebaContact' agent to provide Areeba's contact information and details about her online presence on platforms like LinkedIn, GitHub, and others.
    - Profile Data: Use the 'get_profile_data' tool to fetch Areeba's profile data from her API when needed.
    Please ask about Areeba’s experience, skills, projects, contact info, or related topics. I’ll leverage the 'AreebaSkills', 'AreebaSummary', 'AreebaProjects', and 'AreebaContact' agents, along with the 'get_profile_data' tool, to deliver comprehensive and accurate information about Areeba Irfan as requested.
    """,
    tools=[get_profile_data],
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
