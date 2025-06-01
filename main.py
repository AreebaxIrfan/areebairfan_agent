import os
import chainlit as cl
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
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

# Define the Areeba Irfan Agent with specific instructions
agent1 = Agent(
    instructions=(
        "You are the Areeba Irfan Agent, a helpful assistant restricted to answering questions about: "
        "who Areeba Irfan can contact for professional networking or collaborations, how to arrange a meeting with Areeba Irfan, "
        "what Areeba Irfan does in her free time (e.g., professional development or personal hobbies), "
        "Areeba Irfan’s career history, technical or professional skills, and past or current projects. "
        "Politely decline to answer any questions outside these topics, stating that you are only authorized to provide information "
        "related to Areeba Irfan’s professional connections, skills, projects, and personal interests."
    ),
    name="Areeba Irfan Agent",
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
        "experience", "skill", "skills", "project", "projects", "career", "who", "how", "what"
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
