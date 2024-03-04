from typing import Dict, TypedDict


from typing import TypedDict

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate

from autointerview.utils import reciprocal_rank_fusion

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]



### Nodes ###
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    retriever = state_dict["retriever"]
    model = state_dict["model"]
    #documents = retriever.get_relevant_documents(question)
    
    # RAG-Fusion: Related
    template = """You are a helpful assistant that generates multiple search queries to help a candidate land a job.  These queries will search the candidate's profile for information that best answers the interview question and showcases their suitability for the role. 
    
    Generate multiple search queries related to: {question} 
    Output (4 queries)
    """
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    
    generate_queries = (
        prompt_rag_fusion 
        | ChatOllama(model=model, temperature=0)
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    documents = retrieval_chain_rag_fusion.invoke({"question": question})
    return {"keys": {"documents": documents, "question": question, "model": model}}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    model = state_dict["model"]

    # Prompt
    # prompt = hub.pull("rlm/rag-prompt")
    # Custom AI job search assistant
    template = """You are a helpful job interview assistant. 
    Your goal is to help the candidate excel in their job interview by using information from their profile to answer interview questions.
    Using the candidate's profile, craft a strong answer that highlights their strengths.
    Here is the candidate´s profile:
    
    {context} 
    
    Question: {question} 
    
    Answer: 
    **Important Note:**: 
    Respond as if you are the candidate.
    Make sure to review the answer and ensure it uses "I" statements and reflects the skills and qualities found in the candidate's profile. 
    Create a coherent story.
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # LLM
    llm = ChatOllama(model=model, temperature=0)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "keys": {"documents": documents, "question": question, "generation": generation, "model": model}
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """

    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    model = state_dict["model"]

    # LLM
    llm = ChatOllama(model=model, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=["question","context"],
    )

    # Chain
    chain = prompt | llm | JsonOutputParser()

    # Score
    filtered_docs = []
    for d in documents:
        score = chain.invoke(
            {
                "question": question,
                "context": d,
            }
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    return {"keys": {"documents": filtered_docs, "question": question, "model": model}}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    model = state_dict["model"]

    # LLM
    llm = ChatOllama(model=model, temperature=0)
    
    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n 
        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question:""",
        input_variables=["question"],
    )

    # Chain
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {"keys": {"documents": documents, "question": better_question, "model": model}}


def prepare_for_final_grade(state):
    """
    Passthrough state for final grade.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): The current graph state
    """

    print("---FINAL GRADE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    generation = state_dict["generation"]
    model = state_dict["model"]

    return {
        "keys": {"documents": documents, "question": question, "generation": generation, "model": model}
    }


### Edges ###
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    filtered_documents = state_dict["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents(state):
    """
    Determines whether the generation is grounded in the document.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Binary decision
    """

    print("---GRADE GENERATION vs DOCUMENTS---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    generation = state_dict["generation"]
    model = state_dict["model"]

    # LLM
    llm = ChatOllama(model=model, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=["generation", "documents"],
    )

    # Chain
    chain = prompt | llm | JsonOutputParser()
    score = chain.invoke({"generation": generation, "documents": documents})
    grade = score["score"]

    if grade == "yes":
        print("---DECISION: SUPPORTED, MOVE TO FINAL GRADE---")
        return "supported"
    else:
        print("---DECISION: NOT SUPPORTED, GENERATE AGAIN---")
        return "not supported"


def grade_generation_v_question(state):
    """
    Determines whether the generation addresses the question.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Binary decision
    """

    print("---GRADE GENERATION vs QUESTION---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    generation = state_dict["generation"]
    model = state_dict["model"]

    llm = ChatOllama(model=model, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=["generation", "question"],
    )

    # Prompt
    chain = prompt | llm | JsonOutputParser()
    score = chain.invoke({"generation": generation, "question": question})
    grade = score["score"]

    if grade == "yes":
        print("---DECISION: USEFUL---")
        return "useful"
    else:
        print("---DECISION: NOT USEFUL---")
        return "not useful"