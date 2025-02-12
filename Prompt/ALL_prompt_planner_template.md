# ALL_prompt_planner_template

The following content is part of the prompt used by the PlannerAgent during the training of the BBH task in this program. When using it, please copy the corresponding prompt into the `MARS/Prompt/EDIT_2_prompt_planner_template.txt` file.




## Boolean  Expressions

Split the task '{task_description}' into detailed steps and details. 
For example, for the Boolean Expressions task, the task is planned as follows:
Total steps: 5  
Step 1: Clearly define the task as evaluating the truth value of a Boolean expression using Boolean constants (True, False) and operators (and, or, not).  
Step 2: Specify the input format, ensuring it includes a Boolean expression followed by a delimiter (e.g., "is").  
Step 3: Instruct the model to parse the input, identify the Boolean expression, and evaluate it step by step according to Boolean logic rules.  
Step 4: Emphasize the importance of following operator precedence (not, and, or) and handling parentheses correctly.  
Step 5: Direct the model to output only the final truth value (True or False) without additional explanation or commentary.



## Disambiguation QA

Split the task '{task_description}' into detailed steps and details. 
For example, for the Disambiguation QA task, the task is planned as follows:
Total steps: 4  
Step 1: Identify the pronoun in the sentence and the possible antecedents (nouns it could refer to).  
Step 2: Analyze the context and grammatical structure to determine if the pronoun's antecedent can be inferred.  
Step 3: If the antecedent can be deduced, state the correct noun it refers to; otherwise, declare the sentence as ambiguous.  
Step 4: Match the conclusion with the provided options and select the correct answer.



## Formal Fallacies Syllogisms  Negation

Split the task '{task_description}' into detailed steps and details. 
For example, for the Formal Fallacies Syllogisms Negation task, the task is planned as follows:
Total steps: 4  
Step 1: Clearly define the task and its focus on formal fallacies, syllogisms, and negations.  
Step 2: Structure the prompt to include the context, the argument, and the question about deductive validity.  
Step 3: Ensure the prompt explicitly asks the model to evaluate whether the argument is deductively valid or invalid based on the provided premises.  
Step 4: Format the prompt to include clear instructions and options (e.g., "valid" or "invalid") for the model to choose from.



## Geometric Shapes

Split the task '{task_description}' into detailed steps and details. 
For example, for the Geometric Shapes task, the task is planned as follows:
Total steps: 5  
Step 1: Analyze the requirements of the geometric shape identification task to understand the desired output and the types of shapes to be distinguished.  
Step 2: Examine the example SVG path element to identify patterns, key features, and commands that define different geometric shapes.  
Step 3: Create a clear and structured prompt that instructs the language model to analyze the SVG path element and determine the geometric shape it represents.  
Step 4: Include relevant examples of SVG path elements and their corresponding shapes in the prompt to guide the language model's understanding and reasoning.  
Step 5: Test and refine the prompt iteratively by providing sample SVG paths and adjusting the prompt for clarity, accuracy, and alignment with the correctness goal.  



## Ruin Names

Split the task '{task_description}' into detailed steps and details. 
For example, for the ruin names task, the task is planned as follows:
Total steps: 4  
Step 1: Analyze the task requirements to understand the goal of creating a humorous one-character edit for an artist, band, or movie name.  
Step 2: Identify the key components of the prompt, including the input format, the requirement for humor, and the need for a single-character edit.  
Step 3: Draft a clear and concise prompt that instructs the language model to generate a humorous one-character edit for a given name, ensuring it aligns with the task's objectives.  
Step 4: Review and refine the prompt to maximize clarity and effectiveness in guiding the language model to produce the desired output.



## Sports Understanding

Split the task '{task_description}' into detailed steps and details. 
For example, for the Sports Understanding task, the task is planned as follows:
Total steps: 4  
Step 1: Define the task clearly, specifying that the goal is to evaluate the plausibility of a factitious sentence related to sports based on real-world knowledge of sports events, players, and terminology.  
Step 2: Create a structured prompt that includes an example input and output to guide the model on how to assess plausibility, ensuring it understands the context of sports and common terminology.  
Step 3: Ensure the prompt explicitly instructs the model to analyze the sentence for consistency with known sports facts, player capabilities, and event details, and to output "yes" or "no" based on plausibility.  
Step 4: Test the prompt with multiple examples to verify that the model consistently produces accurate and correct assessments of plausibility for sports-related sentences.