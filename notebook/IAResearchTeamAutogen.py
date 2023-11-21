#pip install pyautogen~=0.2.0b4
import autogen

#get config list which contains the API keys
config_list_gpt = autogen.config_list_from_json(
    "gpt_config.json",
    filter_dict={
        "model": ["gpt-4-1106-preview"],
    },
)
#set the configuration fromt he config list
gpt_config = {"config_list": config_list_gpt}


#Create Autogen Agents--------------------------------------



#human admin who approves plan
user_proxy = autogen.UserProxyAgent(
   name="Admin",
   system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
   code_execution_config=False,
)

#coding assistant agent that can follows plan
engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=gpt_config,
    system_message='''Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
''',
)

#researcher who goes through data
scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config=gpt_config,
    system_message="""Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code."""
)

#planner who creates the plan 
planner = autogen.AssistantAgent(
    name="Planner",
    system_message='''Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist. Once the development step is complete, pass the information over
to the design research team of the UXResearcher and the PartnersAndResources. This team should provide insights specific to the prompt and their expertise. 
''',
    llm_config=gpt_config,
)

#agent that runs code by engineer
executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 3, "work_dir": "paper"},
)

#general critic
critic = autogen.AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
    llm_config=gpt_config,
) 

#user research expert
uxresearcher = autogen.AssistantAgent(
    name="UXResearcher",
    llm_config=gpt_config,
    system_message="""You are a business and design consultant who cares about user experience. 
    You define user wants, needs, and potential for any given question. For you, user experience design begins with determining the most important users 
    and then diving deep to understand their wants and needs. You observe patterns of current activities, workarounds,
     and aspirations, insights are derived to guide the creation of new offerings. Whether insights prescribe entirely new 
     business models or re-tooled elements, a foundational user understanding enables your concepting, design, and mapping of ideal 
     future experiences including offerings, brand, channels, and commerce. Experience strategies also define value within 
     networks of users and partners, ensuring a systemic and user-centered solution. """
)
#partners and resources expert
partnersandresources = autogen.AssistantAgent(
    name="PartnerAndResources",
    llm_config=gpt_config,
    system_message="""You are a business and design consultant who cares about partners and resources. You aim to answer what user needs will be
     served if we leverage others capabilities and profit models. From infrastructure to open innovation, collaboration with complementors or competitors can 
     disrupt a market by quickly enabling access to new expertise, customers, capital, and other resources. Networks can be 
     short term alliances to execute a special project, or enduring partnerships to establish new entities. These relationships
     can reduce cost and risk or enable the creation of otherwise unfeasible offerings. You seek to describe partners and resource elements that  
     tap latent marketplace potential that will deliver lasting user and business value. """
)



#this organizes the chat structure into a data class 
groupchat = autogen.GroupChat(agents=[user_proxy, engineer, scientist, planner, executor, critic], messages=[], max_round=50)
#This sets the response workflow to the chat - still understanding this code
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt_config)

messageRequest = """You are working for Pepsi to help them better understand how their foundation concerned with recylcing
efforts and end-of-life packaging solutions in emerging markets can be improved. You want to help them develop some ideas around end-of-life packaging solutions 
that they should pursue. You also want to help them better understand opportunities to support and gain insights into the foundations they are already funding.
You will start by searching www.mdpi.com for papers related to this topic and summarize findings for their business."""

 

#initiate chat --------------------------------------
user_proxy.initiate_chat(manager, message=messageRequest)

# type exit to terminate the chat
