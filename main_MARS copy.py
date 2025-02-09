import re
from typing import AsyncGenerator, List, Sequence,Tuple
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import AgentMessage, ChatMessage, TextMessage
from autogen_core import CancellationToken
from openai import OpenAI
import pandas as pd
import sys
from datetime import datetime
import os
from tqdm import tqdm  # 导入 tqdm 库
from Agents import TargetAgent,TeacherAgent,StudentAgent,PlannerAgent,CriticAgent,analyze_prompt_history
from Agents import ChatManagerAgent,UserProxyAgent
import Config

Config.current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
file_name = os.path.join('./Output', f'{Config.current_time}_output.txt')
sys.stdout = open(file_name, 'w', encoding='utf-8')


# 运行示例函数
async def run_agents() -> None:
    # 创建各个 Agent 实例
    chat_manager_agent = ChatManagerAgent("chat_manager")

    # 模拟任务输入
    task_message = TextMessage(content="Here is a topic for geometric graph generation, I want to input a prompt and this topic into the big language model so that the big language model outputs the highest correctness rate. Please generate the most suitable prompt according to the requirements I just mentioned", source="user")

    # 先由 ChatManagerAgent 接收任务
    chat_manager_response = await chat_manager_agent.on_messages([task_message], CancellationToken())
    print(chat_manager_response.chat_message.content)


# 使用 asyncio.run 运行示例函数
import asyncio
import time

# 记录程序开始时间
start_time = time.time()
print(f"开始时间: {start_time:.4f} 秒")

asyncio.run(run_agents())

end_time = time.time()
print(f"结束时间: {end_time:.4f} 秒")

run_time = end_time - start_time
print(f"程序运行时间: {run_time:.4f} 秒")

# 恢复标准输出
sys.stdout.close()
sys.stdout = sys.__stdout__