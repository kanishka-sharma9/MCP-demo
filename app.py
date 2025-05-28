import os
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

async def run_mem_chat():
    config_file = "browser_mcp.json"
    print("Initializing Chat...")

    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    agent = MCPAgent(llm=llm, client=client, max_steps=10,system_prompt="""
                    You are a helpful assistant with access to the following tools:\n
                     1. Playright - MCP server for browser automation.
                     2. Airbnb - MCP server for looking up hotels.
                     3. Tavily search - MCP server for web search.\n

                     Use these to complete the tasks assigned to you by the user.
                     Think step-by-step.
                     """)

    try:
        while True:
            user = input("Enter: ")
            if user.lower() in ['exit', 'quit']:
                print("Conversation Ended")
                break
            if user.lower() == 'clear':
                agent.clear_conversation_history()
                continue

            print('\nAssistant: ', flush=True, end="")
            try:
                response = await agent.run(user,max_steps=10)  # Explicitly set max_steps
                print(response)
            except Exception as e:
                print("Agent error:", e)
    except Exception as e:
        print("Main loop error:", e)
    finally:
        if client and client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_mem_chat())