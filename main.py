import os
import chainlit as cl
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner ,function_tool
import requests
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Set up the provider (adjusted base_url for Gemini API)
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/",  # Corrected base URL
)

# Define the model (verify model name with Gemini API)
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",  # Updated to a valid Gemini model
    openai_client=provider,
)

# Run configuration
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)


@function_tool
def getData(data: str) -> str:
    """
    Get the data for a given api of areeba irfan.
    """
    # Replace with your actual weather API URL and key
    result = requests.get(f"https://areebaxirfan.vercel.app/api/profile")
    
    if result.status_code == 200:
        data = result.json()
        return data
    else:
        return "Sorry, I couldn't fetch the data."

areeba_skill :Agent=Agent(
    name="hello",
    instructions="you assist the user to get the data of areeba irfan by using the getData tool , I am a Full Stack Developer skilled in HTML/CSS, JavaScript, and TypeScript, with expertise in building dynamic user interfaces using React, Next.js, and Tailwind CSS. On the backend, I specialize in Python and FastAPI, focusing on efficient API development and scalable architectures. Beyond development, I am passionate about tech teaching, content creation, and technical writing, where I simplify complex topics and share knowledge with the developer community.",
    tools=[getData],
),
summary : Agent= Agent(
    name="summary",
    instructions="""
    Areeba Irfan, a Karachi-based Full Stack Developer and UI/UX expert, is a tech enthusiast with over a year of experience building accessible, user-friendly web applications. Currently pursuing an Associate Degree in Computer Science at Virtual University and an AI program at GIAIC, she creates efficient, scalable code and has completed over 30 projects, including AI-driven tools like MediScan AI Pro, e-commerce platforms like Niky Shoes Website, and Python-based applications like Gemini Chatbot with Streamlit. Her skills span HTML/CSS, JavaScript, TypeScript, React, Next.js, Tailwind CSS, Python, FastAPI, API Development, Full Stack Development, Tech Teaching, Content Creation, and Technical Writing. Areeba has participated in 3+ hackathons, completed challenges like 100 Days of Coding and Ramadan Coding Nights, and earned recognition such as the 1$ Dollar Win. Active on LinkedIn, GitHub, X, Instagram, Facebook, and Medium, she assists peers in coding, speaks English and Urdu, and enjoys coding, writing, and learning.s
    """

),

projects : Agent= Agent(
    name="summary",
    instructions="Areeba Irfan's online presence is accessible through her personal website at https://areebairfan.vercel.app/. She can be contacted via email at the.areebairfan@gmail.com and is active on social platforms including LinkedIn (https://www.linkedin.com/in/areebairfan/), GitHub (https://github.com/AreebaxIrfan), X (https://x.com/areebaXirfan)W, Instagram (https://www.instagram.com/areebaxirfan/), Facebook (https://www.facebook.com/AreebaxIrfan/), and Medium (https://medium.com/@areebaxirfan). Based in Karachi, Pakistan, her location is linked at https://www.google.com/maps/place/pakistan/karachi. Her achievements include the 30 Days 30 Projects Challenge and 100 Days of Coding Challenge, both hosted at https://github.com/areeba-irfan/100-days-of-code, Ramadan Coding Nights at https://github.com/areeba-irfan/ramadan-coding-nights, the 1$ Dollar Win Recognition, and participation in 3+ hackathons (no specific URLs provided for the latter two). Her projects showcase her skills, with notable works like MediScan AI Pro (https://github.com/AreebaxIrfan/GIAIC_Q3/tree/main/%F0%9F%93%82Class_Assignment/assignment_07), AI Chatbot with Chainlit (https://github.com/AreebaxIrfan/GIAIC_Q3/tree/main/Ramadan_Coding_Nights/Day_17_Advance_Agent), Gemini Chatbot with Streamlit (https://github.com/AreebaxIrfan/projects/tree/main/chatbot), Resume Generator (https://github.com/AreebaxIrfan/projects/tree/main/resume_generator), Streamlit Website (https://github.com/AreebaxIrfan/Steamlit-Website), Niky Shoes Website (https://github.com/AreebaxIrfan/Nike_Shoes_Ecommerce_Marketplace), Agentia_World (https://github.com/AreebaxIrfan/agentia_world), Niky Dashboard (https://github.com/AreebaxIrfan/Niky_Dashboard), Bouquet E-commerce Website (https://github.com/AreebaxIrfan/e-commerce), Next.js Admin Dashboard (https://github.com/AreebaxIrfan/next.js-Dashboard), Blog Website with Comments (https://github.com/areeba-irfan/blog-website), Book Hub (https://github.com/AreebaxIrfan/Book-Hub), Personal Portfolio (https://areebairfan.vercel.app/), Random User Generator (https://github.com/areeba-irfan/random-user-generator), Todo List (https://github.com/AreebaxIrfan/to-do-list), Move Cursor (https://github.com/areeba-irfan/move-cursor), Birthday Card (https://github.com/AreebaxIrfan/birthday-card), Niky Clone (https://github.com/AreebaxIrfan/shoes-website), Music Course Website (https://github.com/AreebaxIrfan/music-course-web), Resume Builder (https://github.com/areeba-irfan/resume-builder), Animated Projects (https://github.com/AreebaxIrfan/Animated-Project), Python Projects (https://github.com/AreebaxIrfan/Agentic_AI/tree/main/projects), 100 Days of Coding (https://github.com/areeba-irfan/100-days-of-code), Ramadan Coding Nights (https://github.com/AreebaxIrfan/GIAIC_Q3/tree/main/Ramadan_Coding_Nights), and 30 Days of Projects Coding (https://github.com/areeba-irfan/100-days-of-code)."
),

contact : Agent= Agent(
    name="summary",
    instructions="""
    You can reach me via email at the.areebairfan@gmail.com or connect with me on professional and social platforms including LinkedIn, GitHub, X (Twitter), Instagram, Facebook, and Medium, where I share projects, insights, and technical content.
    """
    
)

# Define the Areeba Irfan Agent with specific instructions
agent1 = Agent(
    instructions=(
        "You are the Areeba Irfan Agent, a helpful assistant restricted to answering questions about Areeba Irfan and you also get data from the getData tool also the summary , contact  , projects , areeba_skill for getting the data of areeba "
    ),
    name="Areeba Irfan Agent",
    tools=[getData],
)

# Function to check if the question is relevant
def is_relevant_question(question: str) -> bool:
    """
    Check if the user's question is related to the allowed topics.
    Uses a keyword-based approach; can be enhanced with NLP if needed.
    """
    question = question.lower()
    relevant_keywords = [
        "contact", "meet", "meeting", "free time", "hobby", "hobbies", "past experience",
        "experience", "skill", "skills", "project", "projects", "career", "who", "how", "what", "hi", "hello", "data",
    ]
    return any(keyword in question for keyword in relevant_keywords)

@cl.on_chat_start
async def handle_chat_start():
    # Initialize chat history
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I'm the Areeba Irfan Agent. How can I assist you?").send()

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
        # Run the agent without streaming
        result = await Runner.run(
            agent1,
            input=history,
            run_config=run_config,
        )

        # Send the complete response
        response = str(result.final_output)  # Ensure response is a string
        await cl.Message(content=response).send()

        # Append the assistant response to history
        history.append({"role": "assistant", "content": response})
        cl.user_session.set("history", history[-10:])  # Limit history to last 10 messages

    except Exception as e:
        error_message = f"Error processing your request: {str(e)}"
        await cl.Message(content=error_message).send()
        history.append({"role": "user", "content": message.content})
        history.append({"role": "assistant", "content": error_message})
        cl.user_session.set("history", history[-10:])  # Limit history
