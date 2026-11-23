import argparse
import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from retriever import guest_info_tool
from tools import DuckDuckGoSearchRun, weather_info_tool, hub_stats_tool

load_dotenv()

search_tool = DuckDuckGoSearchRun()

chat = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))
tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]

chat_with_tools = chat.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }


builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()


example_questions = [
    "Tell me about 'Lady Ada Lovelace'",
    "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?",
    "One of our guests is from Qwen. What can you tell me about their most popular model?",
    "I need to speak with 'Dr. Nikola Tesla' about recent advancements in wireless energy. Can you help me prepare for this conversation?"
]


def save_graph_diagram(path: str):
    """Save a diagram of the agent as a PNG, rendered by the public mermaid.ink service"""
    alfred.get_graph().draw_mermaid_png(output_file_path=path)
    print(f"Graph diagram saved to {Path(path).resolve()}")


def build_langfuse_handler():
    """Return a Langfuse callback when its keys are set, else None"""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def converse(messages, config):
    """Ask Alfred one turn and print the exchange"""
    print(f"\nYou: {messages[-1].content}")
    response = alfred.invoke({"messages": messages}, config=config)
    print(f"\nAlfred's response:\n{response['messages'][-1].content}")
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw", action="store_true", help="save a diagram of the agent as agent_flow.png instead of running the examples")
    args = parser.parse_args()

    if args.draw:
        save_graph_diagram("agent_flow.png")
        return

    langfuse_handler = build_langfuse_handler()
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    if langfuse_handler:
        print("Langfuse tracing is on")

    for question in example_questions:
        converse([HumanMessage(content=question)], config)

    # Conversation memory: the follow-up is answered with the earlier exchange attached.
    first_turn = converse(
        [HumanMessage(content="Tell me about 'Lady Ada Lovelace'. What's her background and how is she related to me?")],
        config,
    )
    converse(
        first_turn["messages"] + [HumanMessage(content="What projects is she currently working on?")],
        config,
    )


if __name__ == "__main__":
    main()
