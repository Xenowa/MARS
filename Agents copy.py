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
import Config





def extract_steps(planner_response: str) -> Tuple[int, List[str]]:
    # 提取步骤数量
    total_steps_match = re.search(r"Total steps: (\d+)", planner_response)
    if not total_steps_match:
        raise ValueError("Planner response does not contain 'Total steps'")
    total_steps = int(total_steps_match.group(1))

    # 提取每一步的描述
    steps = []
    for i in range(1, total_steps + 1):
        step_match = re.search(fr"Step {i}: (.+)", planner_response)
        if step_match:
            steps.append(step_match.group(1).strip())
        else:
            raise ValueError(f"Step {i} is missing in the planner response")

    return total_steps, steps

class ChatManagerAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, "ChatManagerAgent，负责管理调用顺序。")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        print("ChatManagerAgent is called")

        # 先调用 UserProxyAgent
        input_agent = UserProxyAgent("input_agent")
        input_response = await input_agent.on_messages(messages, cancellation_token)
        print("-------------input_response.chat_message.content------------")
        print(input_response.chat_message.content)
        print("-----------------------------------------------------------------")

        # 再调用 PlannerAgent
        planner_agent = PlannerAgent("planner_agent")
        planner_response = await planner_agent.on_messages([input_response.chat_message], cancellation_token)
        print("-------------------planner_response.chat_message.content--------------------")
        print(planner_response.chat_message.content)
        print("-----------------------------------------------------------------")

        # 提取步骤数量和每一步的描述
        total_steps, steps = extract_steps(planner_response.chat_message.content)
        print(f"Total steps: {total_steps}")
        for i, step in enumerate(steps, start=1):
            print(f"Step {i}: {step}")

        # 初始化 TeacherAgent 和 StudentAgent
        teacher_agent = TeacherAgent("teacher_agent")
        student_agent = StudentAgent("student_agent")
        critic_agent = CriticAgent("critic_agent")
        target_agent = TargetAgent("target_agent")

        ####### initial TeacherAgent   Critic Agent  ######
        # ToDo:
        # Teacher 初始化
        teacher_input = TextMessage(
            content=f"Here is the task definition:\n{input_response.chat_message.content}\n",
            source=self.name
        )
        teacher_init = await teacher_agent.on_messages([teacher_input], cancellation_token)

        # Critic 初始化
        critic_init = await critic_agent.on_messages([teacher_init.chat_message], cancellation_token)


        # 初始化 Student 的输入
        student_input = TextMessage(
            content=f"Here is the task definition:\n{input_response.chat_message.content}\n"
                    f"Please generate a more appropriate prompt based on the following prompt and task definition: Think step by step and solve the question.",
            source=self.name
        )
        student_response = await student_agent.on_messages([student_input], cancellation_token)
        print("----------student_response.chat_message.content--------------------")
        print(student_response.chat_message.content)
        print("-----------------------------------------------------------------")

        # 初始化 TargetAgent 的输入
        target_response = await target_agent.on_messages([student_response.chat_message], cancellation_token)
        target_score = target_response.chat_message.content
        print("-------------target_score----------")
        print(target_score)

        # 循环处理每个步骤
        while True:
            # 检查是否满足停止条件
            if target_agent.check_stop_condition():
                print("Stop condition met!")
                break

            # Teacher-Critic-Student circulation
            for step_index, step_description in enumerate(steps, start=1):
                print(f"Processing Step {step_index}: {step_description}")

                # Teacher 提问
                teacher_input = TextMessage(
                    content=f"Here is the task definition:\n{input_response.chat_message.content}\n"
                            f"Here is the prompt given by the student from the previous round:\n{student_response.chat_message.content}\n"
                            f"Ask heuristic questions based on the students' historical responses and the current step: {step_description}\n",
                    source=self.name
                )
                teacher_response = await teacher_agent.on_messages([teacher_input], cancellation_token)
                print("---------teacher_response.chat_message.content-------------------------------------")
                print(teacher_response.chat_message.content)
                print("-----------------------------------------------------------------")

                # critic 评估
                critic_response = await critic_agent.on_messages([teacher_response.chat_message], cancellation_token)
                print("---------critic_response.chat_message.content-------------------------------------")
                print(critic_response.chat_message.content)
                print("-----------------------------------------------------------------")
                
                if "False" in critic_response.chat_message.content:
                    # 重新提问
                    teacher_input = TextMessage(
                        content=f"Here is feedback on whether your output matches the Socratic questioning, please refer to the suggestion to regenerate the questioning:\n{critic_response.chat_message.content}\n"
                                f"Here is the task definition:\n{input_response.chat_message.content}\n"
                                f"Here is the prompt given by the student from the previous round:\n{student_response.chat_message.content}\n"
                                f"Ask heuristic questions based on the students' historical responses and the current step: {step_description}\n",
                        source=self.name
                    )
                    teacher_response = await teacher_agent.on_messages([teacher_input], cancellation_token)
                    print("---------teacher_response.chat_message.content--regenerate-----------------------------------")
                    print(teacher_response.chat_message.content)
                    print("-----------------------------------------------------------------")     

                # 更新 Student 的输入
                student_input = TextMessage(
                    content=f"Here is the task definition:\n{input_response.chat_message.content}\n"
                            f"Here is your last prompt:\n{student_response.chat_message.content}\n"
                            f"Please base on the following question update your prompt:\n{teacher_response.chat_message.content}",
                    source=self.name
                )
                student_response = await student_agent.on_messages([student_input], cancellation_token)
                print("----------student_response.chat_message.content--------------------")
                print(student_response.chat_message.content)
                print("-----------------------------------------------------------------")

            # 更新 TargetAgent 的输入
            target_response = await target_agent.on_messages([student_response.chat_message], cancellation_token)
            target_score = target_response.chat_message.content
            print("-------------target_score----------")
            print(target_score)

        response_message = TextMessage(content="Everything's over!", source=self.name)
        # 在程序运行结束时分析 prompt_history
        analyze_prompt_history(target_agent.prompt_history)

        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

# UserProxyAgent
class UserProxyAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, "负责接收用户输入的内容。")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        # 这里可以添加接收用户输入的逻辑，例如将用户输入转发给其他 Agent 或进行简单处理
        print("UserProxyAgent is called")

        # 打开文件
        with open('./Prompt/EDIT_1_userproxy_task_input.txt', 'r', encoding='utf-8') as file:
            # 读取整个文件内容
            userproxy_task_input= file.read()

        response_message = TextMessage(content=userproxy_task_input, source=self.name)
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass



# 计划者 Agent
class PlannerAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, "负责拆分优化任务为具体步骤和细节。")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        # 假设输入消息包含任务描述，这里使用 ChatGPT-4 API 更智能地拆分任务为步骤和细节
        print("PlannerAgent is called")
        task_description = messages[0].content

        with open("./Prompt/EDIT_2_prompt_planner_template.txt", 'r', encoding='utf-8') as file:
            prompt_planner_template =  file.read()
        prompt_planner = prompt_planner_template.format(task_description=task_description)

        response_text = await self.call_LLM(prompt_planner)

        response_message = TextMessage(content=response_text, source=self.name)
        return Response(chat_message=response_message)

    async def call_LLM(self, prompt):
        # client = OpenAI(api_key="sk-jY5As9om5xq8vsdl8WcLUp6B6iKFCASiKC9YDYH3SaHr2Uvb", base_url="https://api.chatanywhere.tech/v1")
        client = OpenAI(api_key= Config.API_KEY, base_url= Config.BASE_URL)

        # 打开文件
        with open('./Prompt/system_prompt_planner.txt', 'r', encoding='utf-8') as file:
            # 读取整个文件内容
            system_prompt_planner = file.read()

        response = client.chat.completions.create(
            model= Config.MODEL,
            # model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt_planner},
                {"role": "user", "content": prompt}
            ]
            # api_key="sk-jY5As9om5xq8vsdl8WcLUp6B6iKFCASiKC9YDYH3SaHr2Uvb",
            # base_url="https://api.chatanywhere.tech/v1"
        )
        return response.choices[0].message.content

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

# 教师 LLM Agent（体现苏格拉底提问方式）
class TeacherAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, "以苏格拉底提问方式引导学生思考的教师。")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        # 使用 ChatGPT-4 API 根据学生的回答进行苏格拉底式提问
        print("TeacherAgent is called")

        student_response = messages[-1].content
        response_text = await self.call_LLM(student_response)
        response_message = TextMessage(content=response_text, source=self.name)
        return Response(chat_message=response_message)

    async def call_LLM(self, prompt):
        client = OpenAI(api_key=Config.API_KEY, base_url= Config.BASE_URL)

        with open('./Prompt/system_prompt_teacher.txt', 'r', encoding='utf-8') as file:
            # 读取整个文件内容
            system_prompt_teacher = file.read()

        response = client.chat.completions.create(
            model= Config.MODEL,
            messages=[
                {"role": "system", "content": system_prompt_teacher},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

# 学生 LLM Agent（体现苏格拉底的思考方式）
class StudentAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, "以苏格拉底思考方式回答问题的学生。")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        # 使用 ChatGPT-4 API 以苏格拉底思考方式回答问题
        print("StudentAgent is called")
        response_text = await self.call_LLM(messages[0].content)
        response_message = TextMessage(content=response_text, source=self.name)
        return Response(chat_message=response_message)

    async def call_LLM(self, prompt):
        client = OpenAI(api_key=Config.API_KEY, base_url= Config.BASE_URL)

        response = client.chat.completions.create(
            model= Config.MODEL,
            messages=[
                {"role": "system", "content": "You are a prompt generator, please proceed to iterate over the existing prompts as required.\n Note that you should only output the new prompt you generated."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

# 评价者 LLM Agent
class CriticAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, "评价教师提问是否符合苏格拉底方式的评价者。")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        print("CriticAgent is called")
        teacher_question = messages[0].content
        prompt = (
            "Read the following questions posed by the teacher and judge whether the teacher's questioning follows the Socratic style of questioning.\n"
            # "such as whether he or she has guided the students to think in terms of foundations and multiple perspectives.\n"
            f"questions:\n{teacher_question}"
        )
        response_text = await self.call_LLM(prompt)
        response_message = TextMessage(content=response_text, source=self.name)
        return Response(chat_message=response_message)
    
    async def call_LLM(self, prompt):
        client = OpenAI(api_key=Config.API_KEY, base_url= Config.BASE_URL)

        with open('./Prompt/system_prompt_critic.txt', 'r', encoding='utf-8') as file:
            # 读取整个文件内容
            system_prompt_critic = file.read()

        response = client.chat.completions.create(
            model= Config.MODEL,
            messages=[
                {"role": "system", "content": system_prompt_critic},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


# target LLM Agent（用于做题和打分）
class TargetAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, "用于做题和打分的目标 LLM。")
        self.prompt_history = []  # 记录所有测试的 prompt 和正确率
        self.call_count = 0  # 记录被调用的次数

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        print("TargetAgent is called")
        self.call_count += 1  # 每次调用时增加计数器
        ### INPUT DATASET
        dataset = pd.read_csv(Config.DATASET_PATH)
        prompt = messages[0].content
        correct_count = 0
        total_count = len(dataset)

        for index, row in tqdm(dataset.iterrows(), total=total_count, desc="Processing questions"):

        # for index, row in dataset.iterrows():
            question = row['question']
            answer = row['answer']

            # 初始化重试次数
            retry_count = 0
            max_retries = 5  # 最大重试次数

            while retry_count < max_retries:
                # 调用 LLM 进行测试
                # Please don't output the process of doing the question, only the content of the answer. And please do not output any space, period, etc. except the answer.
                #  Please don't output the process of doing the question, only the content of the answer.If the answer consists of only one symbol or word, do not output any space, period, or other characters other than the answer. If the answer consists of a series of words, please follow the requirements of the question.
                #  Please don't output the process of doing the question, only the content of the answer. And please do not output any space, period, etc. except the answer.
                #  The answer should be a parenthesis containing the capital letter of the chosen answer.
                # The answer can only be yes or no
                # Please don't output the process of doing the question, only the content of the answer.The answer should be a parenthesis containing the capital letter of the chosen answer. please do not add any other spaces or symbols.
                target_response = await self.call_LLM_test(prompt + '\nQuestion: ' + question + "\n Please don't output the process of doing the question, only the content of the answer.If the answer consists of only one symbol or word, do not output any space, period, or other characters other than the answer. If the answer consists of a series of words, please follow the requirements of the question.")
                print(f"response:{target_response}")
                print(f"answer:{answer}")
                #  print("--------target_response :-------------")
                # print(target_response)
                # print("---------------------")
                # # 使用正则表达式匹配输出结果
                if target_response == None:
                    generated_answer = " "
                else:
                    generated_answer = target_response  # 直接获取生成的答案
                # 检查生成的答案是否为空
                if generated_answer:
                    break  # 如果生成了答案，退出重试循环
                else:
                    print(f"No answer generated! Retrying... (Retry count: {retry_count + 1})")
                    retry_count += 1
                  #####################################################################
            # #     # 使用正则表达式匹配输出结果
            #     if target_response is None:
            #         print(f"Error: target_response is None. Retrying... (Retry count: {retry_count + 1})")
            #         retry_count += 1
            #         continue  # 继续重试
                
            #     match = re.search(r'\(([A-Z])\)', target_response)
            #     if match:
            #         generated_answer = "(" + match.group(1) + ")"
            #         break  # 匹配成功，退出重试循环
            #     else:
            #         print(f"No format evaluation output! Retrying... (Retry count: {retry_count + 1})")
            #         retry_count += 1     

            ###################################################################
            # 如果重试次数达到上限仍未生成答案，则记录为空答案
            if retry_count == max_retries:
                print(f"Failed to generate answer after {max_retries} retries. Skipping this question.")
                generated_answer = ""
            print(f'Judge:{generated_answer.replace(" ", "").lower() == str(answer).replace(" ", "").lower()}')
            # 检查生成的答案是否正确
            # if generated_answer == str(answer):
            if generated_answer.replace(" ", "").lower() == str(answer).replace(" ", "").lower():
                correct_count += 1

        accuracy = correct_count / total_count
        print(f"Accuracy: {accuracy}")

        # 记录当前的 prompt 和正确率
        self.prompt_history.append((prompt, accuracy))

        # 根据当前时间生成文件名，并保存在 output 文件夹下
        file_name = os.path.join("./Output", f'{Config.current_time}_prompt_accuracy_history.txt')

        # 将 prompt 和正确率写入文件
        with open(file_name, 'a', encoding='utf-8') as file:
            file.write(f"Call Count: {self.call_count}, \nPrompt: {prompt}, \nAccuracy: {accuracy}\n")

        # 返回正确率
        response_message = TextMessage(content=str(accuracy), source=self.name)
        return Response(chat_message=response_message)

    async def call_LLM_test(self, prompt):
        client = OpenAI(api_key=Config.API_KEY, base_url= Config.BASE_URL)

        response = client.chat.completions.create(
            model= Config.MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass
    
    def check_stop_condition(self) -> bool:
        # 检查是否被调用11次
        # return self.call_count >= 10
        return self.call_count >= 2



# 在程序运行结束时，分析 TargetAgent 的历史记录并写入文件
def analyze_prompt_history(prompt_history):
    if not prompt_history:
        print("No prompt history found.")
        return

    # 提取第一次的 prompt 正确率
    first_prompt, first_accuracy = prompt_history[0]

    # 找到正确率最低的一次
    min_accuracy_record = min(prompt_history, key=lambda x: x[1])
    min_prompt, min_accuracy = min_accuracy_record
    min_index = prompt_history.index(min_accuracy_record) + 1  # 第几次迭代

    # 找到正确率最高的一次
    max_accuracy_record = max(prompt_history, key=lambda x: x[1])
    max_prompt, max_accuracy = max_accuracy_record
    max_index = prompt_history.index(max_accuracy_record) + 1  # 第几次迭代


    print("\n\n--- Prompt History Analysis ---\n")
    print(f"First Prompt Accuracy: {first_accuracy}\n")
    print(f"Lowest Accuracy: {min_accuracy} (Iteration: {min_index}, Prompt: {min_prompt})\n")
    print(f"Highest Accuracy: {max_accuracy} (Iteration: {max_index}, Prompt: {max_prompt})\n")
