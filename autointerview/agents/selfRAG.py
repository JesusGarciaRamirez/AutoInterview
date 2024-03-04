from langgraph.graph import END, StateGraph
from autointerview.pipelines.selfRAG import (
retrieve, grade_documents, transform_query, generate, prepare_for_final_grade,
decide_to_generate, grade_generation_v_documents, grade_generation_v_question,
GraphState)


class selfRAG:
    """Dummy container for defining self-RAG graph (in langgraph)
    # ToDo: Explain self-RAG here
    """
    def __init__(self) -> None:
        self.workflow = StateGraph(GraphState)
        self.define_nodes()
        self.define_graph()

    def define_nodes(self):
        # Define the nodes
        self.workflow.add_node("retrieve", retrieve)  # retrieve
        self.workflow.add_node("grade_documents", grade_documents)  # grade documents
        self.workflow.add_node("generate", generate)  # generate
        self.workflow.add_node("transform_query", transform_query)  # transform_query
        self.workflow.add_node("prepare_for_final_grade", prepare_for_final_grade)  # passthrough

    def define_graph(self):
        # Build graph
        self.workflow.set_entry_point("retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        self.workflow.add_edge("transform_query", "retrieve")
        self.workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents,
            {
                "supported": "prepare_for_final_grade",
                "not supported": "generate",
            },
        )
        self.workflow.add_conditional_edges(
            "prepare_for_final_grade",
            grade_generation_v_question,
            {
                "useful": END,
                "not useful": "transform_query",
            },
        )
    def get_agent(self):
        # Check that graph is properly initialized
        if not(self.workflow.nodes):
            raise ValueError("Graph is empty, agent cannot be created yet")
        return self.workflow.compile()
    
