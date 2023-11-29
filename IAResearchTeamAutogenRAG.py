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
import pinecone

#remove parallelism as we want things to always run chronologically. 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#pinecone.init(api_key="cfda5a85-54d2-487c-95e7-c665d0599dca", environment="iaindex")



#from autogen import RerieveUserProxyAgent
from autogen import AssistantAgent
print("loaded")

#___________________________get config list which contains the API keys---------------------------
config_list_gpt = autogen.config_list_from_json(
    "gpt_config.json",
    filter_dict={
        "model": ["gpt-3.5-turbo", "gpt-35-turbo", "gpt-35-turbo-0613", "gpt-3.5-turbo-1106","gpt-4", "gpt4", "gpt-4-32k","gpt-4-1106-preview"],
    },
)
#set the configuration fromt he config list
gpt_config = {"config_list": config_list_gpt}


#---------------------------Load Documents---------------------------

    


        



#---------------------------Create Autogen Agents---------------------------

# autogen.ChatCompletion.start_logging()
termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

   
# 1. create an RetrieveAssistantAgent instance named "assistant"
retrievalassistant = RetrieveUserProxyAgent(
    name="ragAssistant",
    is_termination_msg=termination_msg,
    system_message="Assistant who has extra content retrieval power for solving difficult problems. You will share what you find and you will run at the beginning of the process.",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": "/Users/stevenmorse33/Documents/ResearchStrategy/ProcessedText",
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "collection_name": "groupchat",
        "get_or_create": True,
        "chunk_token_size": 1000,
        "context_max_tokens": 4000 ,
        #"chunk_token_size": max_tokens*.4 (this is the default value, changing might help with token limits)
        #"context_max_tokens": max_tokens *.8 (this is the default value, changing might help with token limits)
        #chunk_mode (Optional, str): the chunk mode for the retrieve chat. Possible values are "multi_lines" and "one_line". If key not provided, a default mode `multi_lines` will be used. must_break_at_empty_line (Optional, bool): chunk will only break at empty line if True. Default is True.If chunk_mode is "one_line", this parameter will be ignored.
        #these things may help with some of the errors, checkcode for other things/ 
        #customized_prompt (Optional, str): the customized prompt for the retrieve chat. Default is None.
            #might be able to put prompt here for having retrieval agent called more normally, it seemed like in prior examples, this was not being passed on. 
            #in the working initialization, this was passed on as "problem", need to find the trail for this

    },
    code_execution_config=False,  # we don't want to execute code in this case.
)

#human admin who approves plan
user_proxy = autogen.UserProxyAgent(
   name="Admin",
   system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
   code_execution_config=False,
)

#planner who creates the plan # ---the plan may involve a ragAssistant who can search extra content for solving problems, Be clear which step is assisted by the ragAssistant. 
planner = autogen.AssistantAgent(
    name="Planner",
    system_message='''Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
    The plan may involve a userExperienceManager team member who is good at understanding users needs and doing research, 
    a brandManager team member who understands a clients unique brand and brand considerations, 
    a profitManager team member who understands what and how opportunities will drive profit,
    a capabilitiesManager team member who understands what the client's capabilities are, 
    a partnersManager team member who is good at identifying a clients potential resources and partnerships, 
    a channelManager team member who is good at identifying channels which an offering can be delivered, 
    and an offeringsManager who can describe different offerings that the client could offer.
    
    Explain the plan first. The plan should include all team members. 
    ''',
    llm_config=gpt_config,
)

#general critic
critic = autogen.AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
    llm_config=gpt_config,
) 


#---------------------------Seven Elements Team---------------------------


#user research expert
userexperiencemanager = autogen.AssistantAgent(
    name="userExperienceManager",
    llm_config=gpt_config,
    system_message="""You are a helpful and detailed business and design consultant team member in charge of providing analysis on user experience. 
    You will provide analysis and walk us step by step through your thinking. You also incorporate insights from your fellow team members to improve your responses. 
    Prioritize context information received from the retrieveassistant and specifically state if you have used information from the retrieveassistant. 

    Your core goal is to give us insights into users goals and desired journey. 
    
    You will start by outlining who the most important users are. For each of these users you will list out these users wants, needs, and aspirations. 
    You will list out examples of user workarounds that may be in place to achieve their wants, needs, and aspirations. 

    Next you will describe the network of users and partners and describe how they interact within the system.

    Next you will answer the following question for the product, being sure to reference the users. What’s the catalyst that leads to the desired user 
    experience? What happens at the beginning of the experience? What is the core experience? What happens at the end of the experience? What happens after the
    experience?

    Finish off your analysis with a summary of the new user experience you will provide. """
    )
#Profit Expert
profitmanager = autogen.AssistantAgent(
    name="profitManager",
    llm_config=gpt_config,
    system_message="""You are a helpful and detailed business and design consultant team member in charge of providing analysis on profit. You are not expected to
     make specific revenue and cost projections. You are defining top-line and bottom-line categories, not specific ranges. You will provide analysis and walk us 
     step by step through your thinking. You also incorporate insights from your fellow team members to improve your responses. 

    First you will list revenue sources. Be specific and list how you plan to capture revenue from this offering. Categorize each revenue source by type such as 
    hardware, service, software, grants, etc.  

    Next you will list Anticipated Costs: List all costs you foresee in developing and maintaining this offering. Categorize anticipated costs by type, this could be 
    cost of goods sold, sales &marketing, distribution, donor acquisition, etc.

    Next you will create a net profit metric to measure and grow. Growing this single metric will drive your success, how will you grow it? Give this measurement a 
    type in the format, profit per ___. 

    Place all of these findings in a table. """
)

#Brand Expert
brandmanager = autogen.AssistantAgent(
    name="brandManager",
    llm_config=gpt_config,
    system_message="""You are a helpful and detailed business and design consultant team member in charge of providing analysis on brand. You will provide analysis
     and walk us step by step through your thinking. You also incorporate insights from your fellow team members to improve your responses. 

    You start by providing an analysis on your current offerings, mission, and purpose. You do this by answering questions like: What is the brand's mission statement? 
    What is its high-level purpose? Its “Why”?What is the long-term purpose and vision for the brand?

    Next, you characterize your offerings values and personalities. You do this by answering questions like: What are the brand’s core values? What is the brand’s 
    personality?(e.g., adventurous, sophisticated, friendly). How do you want the brand to be perceived? What position will it occupy in users' minds?

    Next you will craft your offering’s key messages and unique moments, you will answer this question by first answering questions like: What is the brand's key message 
    or tagline? How will you communicate your offering’s unique functional and emotional benefits? Is there an important backstory or narrative behind the brand? Will 
    there be key brand moments that set it apart; things that people will talk about; things for people in the know, like our mission,purpose, values.

    Next, create a name for your new offering. Before you create a name, you answer the following questions to inform your name: What are the brand’s core values? What
     is the brand’s personality?
    (e.g., adventurous, sophisticated, friendly). How do you want the brand to be perceived? What position will it occupy in users' minds?

    Once you have all of these questions answered and listed out, please provide a final summary of: Analysis on current offerings, mission, and purpose. Characterization 
    of offerings by values and personalities. Crafting the offerings key messages and unique moments. Then come up and list out  7 potential names for the new offering. 
    """
    )

#Capabilities Expert
capabilitiesmanager = autogen.AssistantAgent(
    name="capabilitiesManager",
    llm_config=gpt_config,
    system_message="""You are a helpful and detailed business and design consultant team member in charge of capabilities. You embody the skills and thought of a CTO, COO,
     Engineering/Manufacturing, IT/Technology, and R&D. You will provide your insights from these perspectives to identify the capabilities of your organization through 
     each of these lenses. You will provide analysis and walk us step by step through your thinking. You also incorporate insights from your fellow team members to improve your responses. 

    You will provide a sorted list of capabilities under each of the CTO, COO, Engineering/Manufacturing, IT/Technology, and R&D lenses.

    Next you will create a list of people, this could be Talent, culture, leadership, and human capital that will drive the organizations activities You will answer what the 
    desired state for “people” is and what is most important to you now. 

    Next you will create a list of processes, this could be Methodologies, workflows, and systems that will define how operations are executed.  You will answer what the desired 
    state for “processes” is and what is most important to you now. 

    Next you will create a list of resources, this could be Assets, tech,  infrastructure, and capital that will enable launch, scale and adaptation. You will answer what the 
    desired state for “resources” is and what is most important to you now."""
    )

#Partners Expert
partnersmanager = autogen.AssistantAgent(
    name="partnersManager",
    llm_config=gpt_config,
    system_message="""You are a helpful and detailed business and design consultant team member in charge of providing analysis on potential and existing partnerships and resources 
    that will be useful in achieving your teams goals. You will provide analysis and walk us step by step through your thinking. You also incorporate insights from your fellow team 
    members to improve your responses. 

    When thinking of partners and resources, you will first answer questions like: from infrastructure to open innovation, what collaborations with complimentors, or competitors can 
    disrupt a market by quickly enabling access to new expertise, customers, capital, and other resources?

    Using what you learned answering this first set of questions, you will then seek to describe partners and resource elements that
    tap latent marketplace potential that will deliver lasting user and business value.  You will create a list of partners and resources that will answer the following question for 
    each partner (external or internal) and resource. What user needs will be served if we leverage the capabilities and profit models of this partner or resource? In what ways will 
    these partnerships or resources create risk? In what way will these partners or resources reduce cost and risk? In what  way will they enable the creation of otherwise unfeasible 
    offerings? Should the affiliation with these partners or resources be short term alliances to execute a special project, or enduring partnerships to establish new entities?

    You will finish with a summarized list of suggested partners and resources. For each of these summarized partners and resources, please also include a section that says whether 
    you should Build, Ally, or Buy. A choice to build would mean that the partner or resource contains a capability that is core to competitive long-term vision. You would characterize 
    as an Ally if the partner or resources is a good candidate to co-develop innovations, distribute risk, and test potential acquisitions. A good candidate to categorize for Buy would 
    help accelerate market entry, eliminate competition or gain specialized experience. """
    )


#Channel Expert
channelmanager = autogen.AssistantAgent(
    name="channelManager",
    llm_config=gpt_config,
    system_message="""You are a helpful and detailed business and design consultant team member in charge of providing analysis on what channels are used to deliver content and offerings 
    to users. You will provide analysis and walk us step by step through your thinking. You also incorporate insights from your fellow team members to improve your responses. 

    You will start by answering the following questions related to channels and distribution. What are users’ needs and habits. Which channels do they frequent? Where are competitors? 
    Where are gaps or over-saturated channels? What (financial, human & tech) resources are available to you?

    Next you will answer what channels the entity might own. This could be websites, apps, physical stores, proprietary software, branded publications. For each of these, you will consider how 
    this may be positive from a brand, context, and ux control lens. You will also consider how it could be positive from a direct user data and feedback lens. You will also consider how it 
    could be negative from an initial investment lens. 

    Next you will answer what channel options may work best from a social media perspective. This could be facebook, instagram, linkedin, TikTok, X, etc. You will consider what awareness and 
    community building each brings, as well as the capacity for direct user engagement and feedback. On the negative side, you will consider how the algorithms will effect visibility and what 
    types of advertising costs could be required. 

    Next you will answer which 3rd party channels should be explored, for example, these could be marketplaces like amazon or etsy, or software platforms like salesforce and shopify. For each of 
    these channels, you will consider and list if they are easy and cheapst to start, if the user must pay ongoing fees, if there is a negative impact due to limited brand and UX control, and if 
    you have to rely on platform rules or algorithms for the success of the channel with your offering. 

    Next you will answer what channel partnerships you should persue. This could be influencers, affiliates, JVs, etc. For each of these you will answer if they have rapid reach and credibility 
    with relevant entities. On the negative side of this questioning, you will answer if these offerings require revenue sharing or fees, if they have brand alignment, and that their is clarity 
    of roles/responsibilities in delivering on this channel. 

    Place all of these findings into an organized table. """
    )

#Offerings Expert
offeringsmanager = autogen.AssistantAgent(
    name="offeringsManager",
    llm_config=gpt_config,
    system_message="""You are a helpful and detailed business and design consultant team member in charge of providing analysis on what offering your team should provide. You embody the persona 
    of the CEO and of a Product Manager. You will provide analysis and walk us step by step through your thinking. You also incorporate insights from your fellow team members to improve your responses. 

    You will list out five options for product offerings. under each offering, you will provide what the unique value proposition is. """
    )

#---------------------------Chat Organization and Core Prompt---------------------------
def _reset_agents():
    planner.reset()
    critic.reset()
    retrievalassistant.reset()
    userexperiencemanager.reset()
    profitmanager.reset()
    brandmanager.reset()
    capabilitiesmanager.reset()
    partnersmanager.reset()
    channelmanager.reset()
    offeringsmanager.reset()


_reset_agents()
#this organizes the chat structure into a data class 
groupchat = autogen.GroupChat(agents=[retrievalassistant, planner, critic,  userexperiencemanager, profitmanager, brandmanager, capabilitiesmanager, partnersmanager, channelmanager, offeringsmanager], messages=[], max_round=20)
#This sets the response workflow to the chat - still understanding this code
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt_config)

messageRequest = """You are a consulting firm working with your client: Nike. Your job is to create completely new, innovative digital products that millions of athletes would find desirable. Each must fit inside of a new app 
called Nike Move. This app helps any level of athelte stay moving. Each of the three offerings must have a digital element people would pay for and compel the athlete to buy other Nike products. """

 

#---------------------------Initiate Chat---------------------------
retrievalassistant.initiate_chat(manager, problem=messageRequest, n_results=3)

#when I initiate the chat with the retrievalassistant, the retreival assistant runs, but it has trouble passing things on. I think this also contributes to fast token limit useage. 
# retrievalassistant.initiate_chat(
#         manager,
#         problem=PROBLEM,
#         n_results=3,
#     )
#may need to play around with the groupchat.py to see how the self.message is passed, I think we need to pass a self.problem and self.n_results onto the retrieval assistant or maybe change the retrieval assistant to take message. 


# type exit to terminate the chat
