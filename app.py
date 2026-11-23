import argparse
import os
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()


class EmailState(TypedDict):
    email: Dict[str, Any]

    email_category: Optional[str]

    spam_reason: Optional[str]

    is_spam: Optional[bool]

    email_draft: Optional[str]

    messages: List[Dict[str, Any]]


# gpt-5 models ignore temperature and always run at their default.
model = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"), temperature=0)


def read_email(state: EmailState):
    """Alfred reads and logs the incoming email"""
    email = state["email"]

    print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")

    return {}


def classify_email(state: EmailState):
    """Alfred uses an LLM to determine if the email is spam or legitimate"""
    email = state["email"]

    prompt = f"""
    As Alfred the butler, analyze this email and determine if it is spam or legitimate.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    First, determine if this email is spam. If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).

    Reply in exactly two lines. The first line must be either "spam" or "not spam".
    The second line must be "reason: <why it is spam>" for spam,
    or "category: <category>" for a legitimate email.
    """

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    response_text = response.content.lower()

    # Judge the verdict by the first line only, so wording in the explanation can't flip it.
    verdict, _, details = response_text.strip().partition("\n")
    is_spam = "spam" in verdict and "not spam" not in verdict

    spam_reason = None
    if is_spam and "reason:" in details:
        spam_reason = details.split("reason:")[1].strip()

    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in details:
                email_category = category
                break

    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]

    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }


def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    print(f"Alfred has marked the email as spam. Reason: {state['spam_reason']}")
    print("The email has been moved to the spam folder.")

    return {}


def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    category = state["email_category"] or "general"

    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    This email has been categorized as: {category}

    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """

    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]

    return {
        "email_draft": response.content,
        "messages": new_messages
    }


def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response"""
    email = state["email"]

    print("\n" + "="*50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-"*50)
    print(state["email_draft"])
    print("="*50 + "\n")

    return {}


def route_email(state: EmailState) -> str:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"


email_graph = StateGraph(EmailState)

email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_response", draft_response)
email_graph.add_node("notify_mr_hugg", notify_mr_hugg)

email_graph.add_edge(START, "read_email")
email_graph.add_edge("read_email", "classify_email")

email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {
        "spam": "handle_spam",
        "legitimate": "draft_response"
    }
)

email_graph.add_edge("handle_spam", END)
email_graph.add_edge("draft_response", "notify_mr_hugg")
email_graph.add_edge("notify_mr_hugg", END)

compiled_graph = email_graph.compile()


legitimate_email = {
    "sender": "john.smith@example.com",
    "subject": "Question about your services",
    "body": "Dear Mr. Hugg, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith"
}

spam_email = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100."
}

sample_emails = [
    ("legitimate", legitimate_email),
    ("spam", spam_email)
]


def save_graph_diagram(path: str):
    """Save a diagram of the workflow as a PNG, rendered by the public mermaid.ink service"""
    compiled_graph.get_graph().draw_mermaid_png(output_file_path=path)
    print(f"Graph diagram saved to {Path(path).resolve()}")


def build_langfuse_handler():
    """Return a Langfuse callback when its keys are set, else None"""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw", action="store_true", help="save a diagram of the workflow as mail_flow.png instead of processing emails")
    args = parser.parse_args()

    if args.draw:
        save_graph_diagram("mail_flow.png")
        return

    langfuse_handler = build_langfuse_handler()
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    if langfuse_handler:
        print("Langfuse tracing is on")

    for label, email in sample_emails:
        print(f"\nProcessing {label} email...")
        compiled_graph.invoke({
            "email": email,
            "is_spam": None,
            "spam_reason": None,
            "email_category": None,
            "email_draft": None,
            "messages": []
        }, config=config)


if __name__ == "__main__":
    main()
