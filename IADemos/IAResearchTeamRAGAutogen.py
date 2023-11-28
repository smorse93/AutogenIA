import autogen
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
import chromadb



#from autogen import RerieveUserProxyAgent
from autogen import AssistantAgent
print("loaded")

#___________________________get config list which contains the API keys---------------------------
config_list_gpt = autogen.config_list_from_json(
    "gpt_config.json",
    filter_dict={
        "model": ["gpt-3.5-turbo", "gpt-35-turbo", "gpt-35-turbo-0613", "gpt-4", "gpt4", "gpt-4-32k","gpt-4-1106-preview"],
    },
)
#set the configuration fromt he config list
gpt_config = {"config_list": config_list_gpt}


#---------------------------Load Documents---------------------------

    


        



#---------------------------Create Autogen Agents---------------------------

# autogen.ChatCompletion.start_logging()
termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

   
# 1. create an RetrieveAssistantAgent instance named "assistant"
retrievalAssistant = RetrieveUserProxyAgent(
    name="ragAssistant",
    is_termination_msg=termination_msg,
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
        "chunk_token_size": 1000,
        "model": config_list[0]["model"],
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,  # we don't want to execute code in this case.
)

#human admin who approves plan
user_proxy = autogen.UserProxyAgent(
   name="Admin",
   system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
   code_execution_config=False,
)

#planner who creates the plan 
planner = autogen.AssistantAgent(
    name="Planner",
    system_message='''Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve a ragAssistant who can search extra content for solving problems, a UXResearcher who is good at understanding the needs of users, and a 
PartnersnAndResources team member who is good at understanding a companies partner and resource opportunities. Explain the plan first. Be clear which step is assisted by the ragAssistant. 
Once the development step is complete, pass the information over to the design research team of the UXResearcher and the PartnersAndResources. This team should 
provide insights specific to the prompt and their expertise. 
''',
    llm_config=gpt_config,
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

#---------------------------Chat Organization and Core Prompt---------------------------

#this organizes the chat structure into a data class 
groupchat = autogen.GroupChat(agents=[user_proxy, engineer, scientist, planner, executor, critic], messages=[], max_round=50)
#This sets the response workflow to the chat - still understanding this code
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt_config)

messageRequest = """You are working for Pepsi to help them better understand how their foundation concerned with recylcing
efforts and end-of-life packaging solutions in emerging markets can be improved. You want to help them develop some ideas around end-of-life packaging solutions 
that they should pursue. You also want to help them better understand opportunities to support and gain insights into the foundations they are already funding.
You will start by searching www.mdpi.com for papers related to this topic and summarize findings for their business."""

 

#---------------------------Initiate Chat---------------------------
user_proxy.initiate_chat(manager, message=messageRequest)

# type exit to terminate the chat
