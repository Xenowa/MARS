import re
from typing import AsyncGenerator, List, Sequence,Tuple
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_core import CancellationToken
from openai import OpenAI
import pandas as pd
import sys
from datetime import datetime
import os
from tqdm import tqdm  
import Config





def extract_steps(planner_response: str) -> Tuple[int, List[str]]:
    # Number of extraction steps
    total_steps_match = re.search(r"Total steps: (\d+)", planner_response)
    if not total_steps_match:
        raise ValueError("Planner response does not contain 'Total steps'")
    total_steps = int(total_steps_match.group(1))

    # Extract the description of each step
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
        super().__init__(name, "ChatManagerAgent is responsible for managing the order of calls")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        print("ChatManagerAgent is called")

        #  UserProxyAgent
        input_agent = UserProxyAgent("input_agent")
        input_response = await input_agent.on_messages(messages, cancellation_token)
        print("-------------input_response.chat_message.content------------")
        print(input_response.chat_message.content)
        print("-----------------------------------------------------------------")

        #  PlannerAgent
        planner_agent = PlannerAgent("planner_agent")
        planner_response = await planner_agent.on_messages([input_response.chat_message], cancellation_token)
        print("-------------------planner_response.chat_message.content--------------------")
        print(planner_response.chat_message.content)
        print("-----------------------------------------------------------------")

        # Extract the number of steps and a description of each step
        total_steps, steps = extract_steps(planner_response.chat_message.content)
        print(f"Total steps: {total_steps}")
        for i, step in enumerate(steps, start=1):
            print(f"Step {i}: {step}")

        # initialize
        teacher_agent = TeacherAgent("teacher_agent")
        student_agent = StudentAgent("student_agent")
        critic_agent = CriticAgent("critic_agent")
        target_agent = TargetAgent("target_agent")


        # Teacher Initialization
        teacher_input = TextMessage(
            content=f"Here is the task definition:\n{input_response.chat_message.content}\n",
            source=self.name
        )
        teacher_init = await teacher_agent.on_messages([teacher_input], cancellation_token)

        # Critic Initialization
        critic_init = await critic_agent.on_messages([teacher_init.chat_message], cancellation_token)


        # Student Initialization
        student_input = TextMessage(
            content=f"Here is the task definition:\n{input_response.chat_message.content}\n"
                    f"Please generate a more appropriate prompt based on the following prompt and task definition: Think step by step and solve the question.",
            source=self.name
        )
        student_response = await student_agent.on_messages([student_input], cancellation_token)
        print("----------student_response.chat_message.content--------------------")
        print(student_response.chat_message.content)
        print("-----------------------------------------------------------------")

        # TargetAgent Initialization
        target_response = await target_agent.on_messages([student_response.chat_message], cancellation_token)
        target_score = target_response.chat_message.content
        print("-------------target_score----------")
        print(target_score)

        # Loop through each step
        while True:
            # Check if stopping conditions are met
            if target_agent.check_stop_condition():
                print("Stop condition met!")
                break

            # Teacher-Critic-Student Circulation
            for step_index, step_description in enumerate(steps, start=1):
                print(f"Processing Step {step_index}: {step_description}")

                # Teacher ask
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

                # critic judge
                critic_response = await critic_agent.on_messages([teacher_response.chat_message], cancellation_token)
                print("---------critic_response.chat_message.content-------------------------------------")
                print(critic_response.chat_message.content)
                print("-----------------------------------------------------------------")
                
                if "False" in critic_response.chat_message.content:
                    # re-ask the question
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

                # update Student's input
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

            #  update Critic's input
            target_response = await target_agent.on_messages([student_response.chat_message], cancellation_token)
            target_score = target_response.chat_message.content
            print("-------------target_score----------")
            print(target_score)

        response_message = TextMessage(content="Everything's over!", source=self.name)
        # Analyzing prompt_history
        analyze_prompt_history(target_agent.prompt_history)

        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


class UserProxyAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, "Responsible for receiving user input.")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        print("UserProxyAgent is called")

        with open('./Prompt/EDIT_1_userproxy_task_input.txt', 'r', encoding='utf-8') as file:
            userproxy_task_input= file.read()

        response_message = TextMessage(content=userproxy_task_input, source=self.name)
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass



class PlannerAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, "Responsible for breaking down optimization tasks into specific steps and details.")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        print("PlannerAgent is called")
        task_description = messages[0].content

        with open("./Prompt/EDIT_2_prompt_planner_template.txt", 'r', encoding='utf-8') as file:
            prompt_planner_template =  file.read()
        prompt_planner = prompt_planner_template.format(task_description=task_description)

        response_text = await self.call_LLM(prompt_planner)

        response_message = TextMessage(content=response_text, source=self.name)
        return Response(chat_message=response_message)

    async def call_LLM(self, prompt):
        
        client = OpenAI(api_key= Config.API_KEY, base_url= Config.BASE_URL)


        with open('./Prompt/system_prompt_planner.txt', 'r', encoding='utf-8') as file:
            system_prompt_planner = file.read()

        response = client.chat.completions.create(
            model= Config.MODEL,
            messages=[
                {"role": "system", "content": system_prompt_planner},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


class TeacherAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, "Teachers who guide their students' thinking with Socratic questioning.")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:

        print("TeacherAgent is called")

        student_response = messages[-1].content
        response_text = await self.call_LLM(student_response)
        response_message = TextMessage(content=response_text, source=self.name)
        return Response(chat_message=response_message)

    async def call_LLM(self, prompt):
        client = OpenAI(api_key=Config.API_KEY, base_url= Config.BASE_URL)

        with open('./Prompt/system_prompt_teacher.txt', 'r', encoding='utf-8') as file:
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


class StudentAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, "以苏格拉底思考方式回答问题的学生。")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
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


class CriticAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, "An critic who assesses whether a teacher's questioning is consistent with the Socratic method.")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        print("CriticAgent is called")
        teacher_question = messages[0].content
        prompt = (
            "Read the following questions posed by the teacher and judge whether the teacher's questioning follows the Socratic style of questioning.\n"
            f"questions:\n{teacher_question}"
        )
        response_text = await self.call_LLM(prompt)
        response_message = TextMessage(content=response_text, source=self.name)
        return Response(chat_message=response_message)
    
    async def call_LLM(self, prompt):
        client = OpenAI(api_key=Config.API_KEY, base_url= Config.BASE_URL)

        with open('./Prompt/system_prompt_critic.txt', 'r', encoding='utf-8') as file:
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


class TargetAgent(BaseChatAgent):
    def __init__(self, name: str):
        super().__init__(name, "用于做题和打分的目标 LLM。")
        self.question_type = Config.question_type  # Record the question_type
        self.prompt_history = []
        self.call_count = 0  
    
    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        print("TargetAgent is called")
        self.call_count += 1
        dataset = pd.read_csv(Config.DATASET_PATH)
        # skip the first data
        dataset = dataset.iloc[1:].reset_index(drop=True)

        prompt = messages[0].content
        correct_count = 0
        total_count = len(dataset)

        for index, row in tqdm(dataset.iterrows(), total=total_count, desc="Processing questions"):
            question = row['question']
            answer = row['answer']

            print(f"answer:{answer}")
            
            # Select the processing function according to the type of question
            if self.question_type == "choice":
                generated_answer = await self.process_choice(prompt, question)
            else:
                generated_answer = await self.process_short_answer(prompt, question)

            print(f'Judge:{generated_answer.replace(" ", "").lower() == str(answer).replace(" ", "").lower()}')

            if generated_answer.replace(" ", "").lower() == str(answer).replace(" ", "").lower():
                correct_count += 1

        accuracy = correct_count / total_count
        print(f"Accuracy: {accuracy}")

        # Record to file
        self.prompt_history.append((prompt, accuracy))
        file_name = os.path.join("./Output", f'{Config.current_time}_prompt_accuracy_history.txt')
        with open(file_name, 'a', encoding='utf-8') as file:
            file.write(f"Call Count: {self.call_count}, \nPrompt: {prompt}, \nAccuracy: {accuracy}\n")

        response_message = TextMessage(content=str(accuracy), source=self.name)
        return Response(chat_message=response_message)

    async def process_choice(self, prompt: str, question: str) -> str:
        # print("process choice question!")

        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            target_response = await self.call_LLM_test(
                prompt + '\nQuestion: ' + question + "\n Please don't output the process of doing the question, only the content of the answer. The answer should be a parenthesis containing the capital letter of the chosen answer. Please do not add any other spaces or symbols."
            )
            print(f"response:{target_response}")
            if target_response is None:
                retry_count += 1
                continue  

            match = re.search(r'\(([A-Z])\)', target_response)
            if match:
                return "(" + match.group(1) + ")"

            retry_count += 1

        return ""  # If the maximum number of retries is reached and no result is obtained, the empty string is returned.

    async def process_short_answer(self, prompt: str, question: str) -> str:
        # print("process short answer question!")

        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            target_response = await self.call_LLM_test(
                prompt + '\nQuestion: ' + question + "\n Please don't output the process of doing the question, only the content of the answer."
            )
            print(f"response:{target_response}")

            if target_response:
                return target_response.strip()

            retry_count += 1

        return ""  # If the maximum number of retries is reached and no result is obtained, the empty string is returned.


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
        # condition 1
        if self.call_count >= 10:
            return True
        
        # conition2: The recent accuracy change is below the threshold.
        if len(self.prompt_history) >= 2:
            last_accuracy = self.prompt_history[-1][1]
            second_last_accuracy = self.prompt_history[-2][1]
            accuracy_diff = abs(last_accuracy - second_last_accuracy)
            
            if accuracy_diff < 0.01:
                return True
        
        return False



# At the end of the program run, analyze TargetAgent's history and write it to a file.
def analyze_prompt_history(prompt_history):
    if not prompt_history:
        print("No prompt history found.")
        return

    # first accuracy
    first_prompt, first_accuracy = prompt_history[0]

    # lowest accuracy
    min_accuracy_record = min(prompt_history, key=lambda x: x[1])
    min_prompt, min_accuracy = min_accuracy_record
    min_index = prompt_history.index(min_accuracy_record) + 1  # Iteration number

    # highest acuracy
    max_accuracy_record = max(prompt_history, key=lambda x: x[1])
    max_prompt, max_accuracy = max_accuracy_record
    max_index = prompt_history.index(max_accuracy_record) + 1  # Iteration number


    print("\n\n--- Prompt History Analysis ---\n")
    print(f"First Prompt Accuracy: {first_accuracy}\n")
    print(f"Lowest Accuracy: {min_accuracy} (Iteration: {min_index}, Prompt: {min_prompt})\n")
    print(f"Highest Accuracy: {max_accuracy} (Iteration: {max_index}, Prompt: {max_prompt})\n")
