import re
from typing import AsyncGenerator, List, Sequence,Tuple
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_core import CancellationToken
import sys
from datetime import datetime
import os
from Agents import TargetAgent,TeacherAgent,StudentAgent,PlannerAgent,CriticAgent,analyze_prompt_history
from Agents import ChatManagerAgent,UserProxyAgent
import Config
import asyncio
import time

def get_question_type():
    """ Read command line arguments and return the question type """
    if len(sys.argv) > 1:
        return sys.argv[1]
    return "choice"  # Default Choice Questions

async def run_agents():
    """ Run Agents, passing the question type """
    chat_manager_agent = ChatManagerAgent("chat_manager")
    target_agent = TargetAgent("target_agent") 

    task_message = TextMessage(
        content="Here is a topic for geometric graph generation, I want to input a prompt and this topic into the big language model so that the big language model outputs the highest correctness rate.",
        source="user"
    )

    chat_manager_response = await chat_manager_agent.on_messages([task_message], CancellationToken())
    print(chat_manager_response.chat_message.content)

if __name__ == "__main__":

    Config.current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = os.path.join('./Output', f'{Config.current_time}_output.txt')
    sys.stdout = open(file_name, 'w', encoding='utf-8')
    
    Config.question_type = get_question_type()
    start_time = time.time()
    print(f"start time: {start_time:.4f} s")

    asyncio.run(run_agents())

    end_time = time.time()
    print(f"end time: {end_time:.4f} s")
    print(f"program runtime: {end_time - start_time:.4f} s")

    # Restore standard output
    sys.stdout.close()
    sys.stdout = sys.__stdout__
