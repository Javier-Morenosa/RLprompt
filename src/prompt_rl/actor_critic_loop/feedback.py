from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    import gradio as gr
    _HAS_GRADIO = True
except ImportError:
    _HAS_GRADIO = False


@dataclass
class MultiSelectResult:
    selected_indices: list[int]
    weights: list[float]


def compute_rewards_from_selection(
    selected_indices: list[int],
    num_total: int,
) -> list[float]:
    rewards = [0.0] * num_total
    for idx in selected_indices:
        if 0 <= idx < num_total:
            rewards[idx] = 1.0
    return rewards


def _run_multiselect_app(
    task_queue: queue.Queue,
    result_queue: queue.Queue,
    server_name: str = "0.0.0.0",
    server_port: Optional[int] = None,
) -> None:
    if not _HAS_GRADIO:
        raise ImportError("Gradio required: pip install prompt-rl[gradio]")
    current = {"query": "", "responses": []}

    def load_next() -> tuple[Any, str, str]:
        try:
            item = task_queue.get(timeout=0.5)
            if isinstance(item, dict):
                q = item.get("query", "")
                resps = item.get("responses", [])
                current["query"] = q
                current["responses"] = resps
                resp_text = "\n\n".join(f"**[{i+1}]** {r}" for i, r in enumerate(resps))
                choices = [str(i + 1) for i in range(len(resps))]
                return gr.update(choices=choices, value=[]), q, resp_text
        except queue.Empty:
            pass
        q = current["query"]
        resps = current["responses"]
        resp_text = "\n\n".join(f"**[{i+1}]** {r}" for i, r in enumerate(resps))
        choices = [str(i + 1) for i in range(len(resps))]
        return gr.update(choices=choices), q, resp_text

    def submit_selection(selected: list[str]) -> str:
        indices = []
        for s in (selected or []):
            try:
                n = int(s)
                if 1 <= n <= len(current["responses"]):
                    indices.append(n - 1)
            except ValueError:
                pass
        result_queue.put({"selected_indices": indices})
        n = len(current["responses"])
        return f"**Submitted.** {len(indices)} correct (of {n}). More correct = higher Actor reward. Critic will use this to refine."

    with gr.Blocks(title="Actor-Critic: Human Feedback", css=".feedback-container { max-width: 900px; }") as app:
        gr.Markdown("""## Actor-Critic Feedback
**Select ALL responses you consider correct.** The more correct, the higher the Actor's reward. The Critic uses only these rewards (never the responses) to guide refinement.""")
        with gr.Row():
            load_btn = gr.Button("Load next task", variant="primary")
        with gr.Row():
            query_box = gr.Textbox(label="Query", lines=2, interactive=False)
        with gr.Row():
            responses_box = gr.Markdown("Responses will appear after Load next task.")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Check each correct response:**")
                selection_checkboxes = gr.CheckboxGroup(choices=[], label="Correct responses")
        with gr.Row():
            submit_btn = gr.Button("Submit selection", variant="secondary")
        out_msg = gr.Markdown("")
        load_btn.click(
            fn=load_next,
            inputs=[],
            outputs=[selection_checkboxes, query_box, responses_box],
        )
        submit_btn.click(
            fn=submit_selection,
            inputs=[selection_checkboxes],
            outputs=[out_msg],
        )
    app.queue()
    app.launch(server_name=server_name, server_port=server_port or 7861)


class HumanMultiSelectFeedback:
    def __init__(
        self,
        server_name: str = "0.0.0.0",
        server_port: Optional[int] = None,
        callback: Optional[Callable[[str, list[str]], list[int]]] = None,
    ) -> None:
        self._server_name = server_name
        self._server_port = server_port
        self._callback = callback
        self._task_queue: queue.Queue = queue.Queue()
        self._result_queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._thread = threading.Thread(
            target=_run_multiselect_app,
            args=(self._task_queue, self._result_queue),
            kwargs={"server_name": self._server_name, "server_port": self._server_port},
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def get_selection(
        self,
        query: str,
        responses: list[str],
    ) -> MultiSelectResult:
        if self._callback:
            indices = self._callback(query, responses)
            return MultiSelectResult(selected_indices=indices, weights=[1.0] * len(indices))
        if not self._started:
            self.start()
        self._task_queue.put({"query": query, "responses": responses})
        result = self._result_queue.get()
        indices = result.get("selected_indices", [])
        return MultiSelectResult(selected_indices=indices, weights=[1.0] * len(indices))


def launch_standalone(
    server_name: str = "127.0.0.1",
    server_port: Optional[int] = None,
    model: str = "gemma3:1b",
    base_url: str = "http://localhost:11434/v1",
    use_mock: bool = False,
) -> None:
    from prompt_rl.llm import MockLLM
    from prompt_rl.actor_critic_loop.actor import LLMActor
    from prompt_rl.actor_critic_loop.critic import LLMCritic
    if use_mock:
        llm = MockLLM(default_response="Sample response.")
    else:
        try:
            from prompt_rl.llm import LocalLLMBackend
            llm = LocalLLMBackend(model=model, base_url=base_url)
            print(f"Using Ollama: {model}")
        except ImportError:
            llm = MockLLM(default_response="Sample response.")
            print("Install prompt-rl[openai] for real LLM. Using MockLLM.")
    actor = LLMActor(prompt_llm=llm, response_llm=llm, max_tokens=256, temperature=0.8)
    critic = LLMCritic(llm=llm, max_tokens=256, temperature=0.3)
    launch_integrated(
        actor=actor,
        critic=critic,
        base_instruction="You are a helpful assistant.",
        num_variations=10,
        server_name=server_name,
        server_port=server_port,
    )


def launch_integrated(
    actor: Any,
    critic: Any,
    base_instruction: str = "You are a helpful assistant.",
    num_variations: int = 10,
    server_name: str = "127.0.0.1",
    server_port: Optional[int] = None,
) -> None:
    if not _HAS_GRADIO:
        raise ImportError("Gradio required: pip install prompt-rl[gradio]")
    from prompt_rl.actor_critic_loop.critic import CriticInput
    state = {"refinement_hint": None, "prev_total": None, "outputs": [], "query": ""}

    def generate_responses(query: str) -> tuple[str, str, Any, str]:
        if not query or not query.strip():
            return "", "Enter a question.", gr.update(choices=[], value=[]), ""
        q = query.strip()
        state["query"] = q
        prompts = actor.generate_prompt_variations(
            base_instruction,
            num_variations,
            refinement_hint=state["refinement_hint"],
            query=q,
        )
        outputs = actor.generate_responses(prompts, q)
        state["outputs"] = outputs
        resp_text = "\n\n".join(f"**[{i+1}]** {o.response}" for i, o in enumerate(outputs))
        choices = [str(i + 1) for i in range(len(outputs))]
        return q, resp_text, gr.update(choices=choices, value=[]), "10 responses generated. Select which are correct."

    def submit_selection(selected: list[str], human_comment: str) -> tuple[str, str, Any, str, str]:
        outputs = state.get("outputs", [])
        if not outputs:
            return "Generate responses first.", "", gr.update(choices=[], value=[]), "", ""
        indices = []
        for s in (selected or []):
            try:
                n = int(s)
                if 1 <= n <= len(outputs):
                    indices.append(n - 1)
            except ValueError:
                pass
        rewards = [1.0 if i in indices else 0.0 for i in range(len(outputs))]
        comment = (human_comment or "").strip()
        has_comment = bool(comment)
        critic_inputs = [
            CriticInput(system_prompt=o.system_prompt, reward=rewards[o.index], has_feedback_comment=has_comment, human_comment=comment if has_comment else None)
            for o in outputs
        ]
        refinement = critic.get_refinement_direction(critic_inputs, context={"previous_total_correct": state.get("prev_total")})
        state["refinement_hint"] = refinement
        state["prev_total"] = len(indices)
        refinement_msg = f"**{len(indices)} correct.** Critic refinement:\n\n{refinement}"
        q = state.get("query", "")
        if not q:
            return refinement_msg, "", gr.update(choices=[], value=[]), "", ""
        prompts = actor.generate_prompt_variations(base_instruction, num_variations, refinement_hint=state["refinement_hint"], query=q)
        new_outputs = actor.generate_responses(prompts, q)
        state["outputs"] = new_outputs
        resp_text = "\n\n".join(f"**[{i+1}]** {o.response}" for i, o in enumerate(new_outputs))
        choices = [str(i + 1) for i in range(len(new_outputs))]
        return refinement_msg, resp_text, gr.update(choices=choices, value=[]), "New iteration: 10 responses regenerated. Select which are correct.", ""

    with gr.Blocks(title="Actor-Critic") as app:
        gr.Markdown("""## Actor-Critic: Question & Feedback
**1.** Enter your question and click **Submit**. The Student LLM will generate 10 system prompt variations and 10 responses.
**2.** Select ALL correct responses, optionally add a comment, then **Submit selection**. A new iteration runs automatically with Actor and Critic updated from your feedback. Repeat until satisfied.""")
        with gr.Row():
            question_input = gr.Textbox(label="Your question", placeholder="e.g. What is machine learning?", lines=2)
        submit_btn = gr.Button("Submit (generates 10 LLM responses)", variant="primary")
        status_msg = gr.Markdown("")
        with gr.Row():
            query_disp = gr.Textbox(label="Query", interactive=False)
        with gr.Row():
            responses_disp = gr.Markdown("Responses generated by the Student LLM will appear here after Submit.")
        with gr.Row():
            checkboxes = gr.CheckboxGroup(choices=[], label="Select ALL correct responses")
        with gr.Row():
            human_comment_box = gr.Textbox(
                label="Optional feedback comment (visible to Critic)",
                placeholder="e.g. Prefer shorter answers, avoid X...",
                lines=2,
            )
        submit_selection_btn = gr.Button("Submit selection", variant="secondary")
        refinement_out = gr.Markdown("")

        submit_btn.click(
            fn=generate_responses,
            inputs=[question_input],
            outputs=[query_disp, responses_disp, checkboxes, status_msg],
        )
        submit_selection_btn.click(
            fn=submit_selection,
            inputs=[checkboxes, human_comment_box],
            outputs=[refinement_out, responses_disp, checkboxes, status_msg, human_comment_box],
        )
    app.queue()
    app.launch(server_name=server_name, server_port=server_port or 7861)
