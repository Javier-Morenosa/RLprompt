"""
Example: Human feedback via Gradio UI.

Run standalone (manual query/response):
  pip install prompt-rl[gradio]
  python examples/gradio_feedback_example.py

Or use HumanFeedbackCollector in your training loop to block until the user
submits feedback for each (query, response) pair.
"""

def main_standalone() -> None:
    """Launch the Gradio app; user enters query/response and submits ratings."""
    try:
        from prompt_rl.feedback import launch_standalone
    except ImportError:
        print("Install Gradio first: pip install prompt-rl[gradio]")
        return
    launch_standalone(server_name="127.0.0.1", server_port=7860)


def main_with_collector() -> None:
    """
    Example: training loop pushes (query, response) to the UI and blocks
    until the user submits feedback. Start the script, then in the browser
    click "Load next task" to fetch the item, rate it, and "Submit feedback".
    """
    try:
        from prompt_rl.feedback import HumanFeedbackCollector
    except ImportError:
        print("Install Gradio first: pip install prompt-rl[gradio]")
        return

    collector = HumanFeedbackCollector(server_name="127.0.0.1", server_port=7860)
    collector.start()
    print("Gradio UI is starting. Open the URL in your browser.")
    print("When prompted, click 'Load next task', then rate and 'Submit feedback'.\n")

    # Simulate one feedback request
    result = collector.get_feedback(
        query="What is machine learning?",
        response="Machine learning is a subset of AI that enables systems to learn from data.",
    )
    print(f"Received score: {result.score:.2f} (thumbs_up={result.thumbs_up}, rating={result.rating})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "collector":
        main_with_collector()
    else:
        main_standalone()
