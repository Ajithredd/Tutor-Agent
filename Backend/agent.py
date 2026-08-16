import json
# pyrefly: ignore [missing-import]
from langchain.agents import create_agent
# pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
# pyrefly: ignore [missing-import]
from langgraph.types import Command
# pyrefly: ignore [missing-import]
from langgraph.types import interrupt
from tools import db_update_student_profile, db_get_student_profile
# pyrefly: ignore [missing-import]
from langchain.agents.middleware import AgentMiddleware, ModelRetryMiddleware, ToolRetryMiddleware, ModelCallLimitMiddleware
from tools import model, TUTOR_TOOLS, db_conn
# pyrefly: ignore [missing-import]
from langchain_core.runnables import RunnableConfig
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict, Any
# pyrefly: ignore [missing-import]
from langgraph.graph.message import add_messages

SYSTEM_PROMPT = """You are an encouraging, expert AI tutor.

Follow these strict interaction rules based on the student's message:

1. GREETINGS & INTROS (e.g. "hi", "hey", "hello"):
   - Greet the student warmly and ask what subject or topic they want to learn today (e.g., Math, Science, Python, History).
   - DO NOT call any tools on simple greetings.

2. EXPLAINING A CONCEPT:
   - When a student asks about a topic or chooses a subject, first invoke `explain_concept` to provide a clear, concise explanation.
   - After explaining, offer to test their understanding with a quiz question.

3. GENERATING A QUIZ:
   - Invoke `generate_quiz_question` ONLY after explaining a concept or when the student explicitly requests a quiz question on a specific topic.
   - You MUST adapt the difficulty based on recent performance:
     * Check the conversation history or call `get_mastery_scores` to see the student's performance on the topic.
     * If they got the last question incorrect, call `generate_quiz_question` with `difficulty="easy"`. You MUST explicitly reference this in your response (e.g. "You got the last one wrong, let's try an easier version to practice." or similar encouraging words).
     * If they got the last question correct, challenge them with a "medium" or "hard" question.

4. GRADING A STUDENT ANSWER (e.g. "I choose answer option A", "A", "option B", or any answer to a quiz):
   - You MUST look at the active quiz question in the conversation history.
   - First, invoke the `grade_answer` tool passing:
     * `question`: the exact question text from the active quiz question.
     * `correct_answer`: the correct answer from that quiz question.
     * `student_answer`: the student's response.
   - Once the `grade_answer` tool execution returns the result:
     * You MUST immediately invoke `update_mastery_score` passing the current topic and the `correct` boolean from the grading result.
   - CRITICAL: DO NOT call `generate_quiz_question` or `explain_concept` in the same turn while grading.
   - CRITICAL: DO NOT claim there is a misunderstanding or confusion. Grade the student's answer first!
   - After `update_mastery_score` completes, provide encouraging feedback based on the grade, and ask the student if they would like another practice question or want to explore a new topic.

5. GENERAL GUIDELINES:
   - Never reveal the correct answer before the student makes an attempt.
   - Keep your text responses concise, encouraging, and conversational.
"""

# Phase 6 Custom Middlewares
class GradingLoggingMiddleware(AgentMiddleware):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def _log_grading(self, request, result):
        if not self.enabled:
            return
        if request.tool_call["name"] == "grade_answer":
            thread_id = request.runtime.config.get("configurable", {}).get("thread_id", "default")
            args = request.tool_call.get("args", {})
            question = args.get("question", "")
            
            try:
                content_str = result.content if hasattr(result, "content") else str(result)
                output_dict = json.loads(content_str)
                is_correct = output_dict.get("correct", False)
                
                topic = "General"
                messages = request.state.get("messages", [])
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            if tc["name"] == "generate_quiz_question":
                                topic = tc["args"].get("topic", "General")
                                break
                        if topic != "General":
                            break
                
                weak_spot = None
                if not is_correct:
                    weak_spot = question[:60] + "..." if len(question) > 60 else question
                    
                db_update_student_profile(thread_id, topic, is_correct, weak_spot)
                print(f"[GradingLoggingMiddleware] Grade logged: thread_id={thread_id}, topic={topic}, correct={is_correct}")
            except Exception as ex:
                print(f"[GradingLoggingMiddleware] Logging failed: {ex}")

    def wrap_tool_call(self, request, handler):
        result = handler(request)
        self._log_grading(request, result)
        return result

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        self._log_grading(request, result)
        return result

class AnswerGuardrailMiddleware(AgentMiddleware):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def _guardrail_check(self, request, response):
        if not self.enabled:
            return response
            
        messages = request.state.get("messages", [])
        last_quiz_correct_answer = None
        student_has_answered = False
        
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                student_has_answered = True
            if isinstance(msg, ToolMessage) and msg.name == "generate_quiz_question":
                try:
                    quiz_data = json.loads(msg.content)
                    last_quiz_correct_answer = quiz_data.get("correct_answer")
                except Exception:
                    pass
                break
        
        if last_quiz_correct_answer and not student_has_answered:
            content = response.content if hasattr(response, "content") else str(response)
            reveal_patterns = [
                f"correct answer is {last_quiz_correct_answer}",
                f"correct option is {last_quiz_correct_answer}",
                f"answer is {last_quiz_correct_answer}",
                f"choose {last_quiz_correct_answer}",
                f"option {last_quiz_correct_answer} is correct"
            ]
            if any(pat in content.lower() for pat in reveal_patterns):
                print(f"[AnswerGuardrailMiddleware] Answer reveal blocked!")
                response.content = "I cannot give away the answer yet! Please try answering the question first."
                
        return response

    def wrap_model_call(self, request, handler):
        response = handler(request)
        return self._guardrail_check(request, response)

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        return self._guardrail_check(request, response)

class HITLInterruptMiddleware(AgentMiddleware):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def wrap_tool_call(self, request, handler):
        result = handler(request)
        if not self.enabled:
            return result
            
        if request.tool_call["name"] == "generate_quiz_question":
            print(f"[HITLInterruptMiddleware] Pausing for student answer...")
            student_answer = interrupt({
                "action": "wait_for_student_answer",
                "question": result.content if hasattr(result, "content") else str(result)
            })
            print(f"[HITLInterruptMiddleware] Resumed. Answer: {student_answer}")
            
            tool_msg = ToolMessage(
                content=result.content if hasattr(result, "content") else str(result),
                name=request.tool_call["name"],
                tool_call_id=request.tool_call["id"]
            )
            human_msg = HumanMessage(content=str(student_answer))
            return Command(update={"messages": [tool_msg, human_msg]})
            
        return result

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        if not self.enabled:
            return result
            
        if request.tool_call["name"] == "generate_quiz_question":
            print(f"[HITLInterruptMiddleware] Pausing for student answer...")
            student_answer = interrupt({
                "action": "wait_for_student_answer",
                "question": result.content if hasattr(result, "content") else str(result)
            })
            print(f"[HITLInterruptMiddleware] Resumed. Answer: {student_answer}")
            
            tool_msg = ToolMessage(
                content=result.content if hasattr(result, "content") else str(result),
                name=request.tool_call["name"],
                tool_call_id=request.tool_call["id"]
            )
            human_msg = HumanMessage(content=str(student_answer))
            return Command(update={"messages": [tool_msg, human_msg]})
            
        return result

# ──────────────────────────────────────────────
# Greeting Detection
# ──────────────────────────────────────────────
_GREETING_WORDS = frozenset({
    "hi", "hey", "hello", "good morning", "good evening", "howdy",
    "start", "restart", "reset", "new", "new chat",
})

def _is_greeting(text: str) -> bool:
    """Return True if *text* is a simple greeting / session-restart request."""
    cleaned = text.strip().lower().rstrip("!.,?")
    if cleaned in _GREETING_WORDS:
        return True
    words = cleaned.split()
    if words and words[0] in {"hi", "hey", "hello"} and len(words) <= 3:
        return True
    return False


# ──────────────────────────────────────────────
# Graph State
# ──────────────────────────────────────────────
class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    curriculum: list[str]
    current_lesson_index: int
    needs_quiz: bool


# ──────────────────────────────────────────────
# Build the Multi-Agent Graph
# ──────────────────────────────────────────────
def build_agent(
    checkpointer,
    model_retry_enabled=True,
    tool_retry_enabled=True,
    limit_enabled=True,
    guardrail_enabled=True,
    grading_log_enabled=True,
    hitl_enabled=True,
):
    # ── Shared middleware stack ──
    middleware_common = []
    if model_retry_enabled:
        middleware_common.append(ModelRetryMiddleware(max_retries=3))
    if tool_retry_enabled:
        middleware_common.append(ToolRetryMiddleware(max_retries=3))
    if limit_enabled:
        middleware_common.append(ModelCallLimitMiddleware(run_limit=10, exit_behavior="error"))

    # ── Sub-Agent 1: Greeter ──
    greeter_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=SystemMessage(content=(
            "You are an encouraging, expert AI tutor.\n"
            "The student has just greeted you or started a new session.\n"
            "Greet them warmly and ask what subject or topic they would like to learn today "
            "(for example: AI Agents, Python, Biology, Operating Systems, Math, etc.).\n"
            "Do NOT teach anything or ask quiz questions. Simply welcome them and ask for their topic."
        )),
        middleware=middleware_common,
    )

    # ── Sub-Agent 2: Curriculum Planner ──
    planner_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=SystemMessage(content=(
            "You are a master curriculum planner.\n"
            "The student has chosen a topic to learn.\n"
            "Acknowledge their choice enthusiastically and introduce a clear, structured "
            "3-lesson curriculum.\n"
            "Format your response as:\n"
            '"Great choice! Here\'s your learning plan for **[Topic]**:\n\n'
            "📚 Lesson 1: [Title] — [1-line description]\n"
            "📚 Lesson 2: [Title] — [1-line description]\n"
            "📚 Lesson 3: [Title] — [1-line description]\n\n"
            'Let\'s dive into Lesson 1!"'
        )),
        middleware=middleware_common,
    )

    # ── Sub-Agent 3: Tutor (RAG-grounded) ──
    from tools import explain_concept, retrieve_reference
    tutor_agent = create_agent(
        model=model,
        tools=[explain_concept, retrieve_reference],
        system_prompt=SystemMessage(content=(
            "You are an expert tutor.\n"
            "Clearly explain and teach the current lesson concept.\n"
            "Use `explain_concept` or `retrieve_reference` to ground your explanation.\n"
            "Use intuitive analogies and real-world examples.\n"
            "At the end, say: \"Ready to test your understanding with a quick quiz?\""
        )),
        middleware=middleware_common,
    )

    # ── Sub-Agent 4: Examiner ──
    from tools import generate_quiz_question, grade_answer
    middleware_examiner = list(middleware_common)
    middleware_examiner.append(AnswerGuardrailMiddleware(enabled=guardrail_enabled))
    middleware_examiner.append(GradingLoggingMiddleware(enabled=grading_log_enabled))
    middleware_examiner.append(HITLInterruptMiddleware(enabled=hitl_enabled))

    examiner_agent = create_agent(
        model=model,
        tools=[generate_quiz_question, grade_answer],
        system_prompt=SystemMessage(content=(
            "You are an examiner agent.\n"
            "Call `generate_quiz_question` to create a quiz on the current lesson.\n"
            "When the student answers (e.g. 'I choose answer option A'), "
            "call `grade_answer` to evaluate.\n"
            "If correct, congratulate them. If wrong, explain the correct answer encouragingly."
        )),
        middleware=middleware_examiner,
    )

    # ═════════════════════════════════════════
    #  GRAPH NODES
    # ═════════════════════════════════════════

    def greeter_node(state: MultiAgentState, config: RunnableConfig):
        """Welcome the student and reset all curriculum state."""
        res = greeter_agent.invoke(state, config)
        return {
            "messages": res.get("messages", []),
            "topic": "",
            "curriculum": [],
            "current_lesson_index": 0,
            "needs_quiz": False,
        }

    def planner_node(state: MultiAgentState, config: RunnableConfig):
        """Create a 3-lesson curriculum from the student's chosen topic."""
        messages = state.get("messages", [])
        user_topic = next(
            (msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)),
            "AI Agents",
        )

        res = planner_agent.invoke(state, config)

        # Parse the 3 lesson titles from the planner's free-text output
        curriculum: list[str] = []
        try:
            planner_text = res.get("messages", [])[-1].content
            parse_prompt = (
                f"Extract ONLY the 3 lesson titles from this curriculum text as a JSON list:\n"
                f'"""{planner_text}"""\n'
                f'Return ONLY a valid JSON array of 3 strings. Example: ["Variables", "Loops", "Functions"]'
            )
            parse_res = model.invoke(parse_prompt)
            content = parse_res.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content.strip())
            if isinstance(parsed, list) and len(parsed) >= 2:
                curriculum = parsed[:3]
        except Exception:
            pass

        if not curriculum:
            curriculum = [
                f"{user_topic} — Fundamentals",
                f"{user_topic} — Core Concepts",
                f"{user_topic} — Advanced Applications",
            ]

        return {
            "messages": res.get("messages", []),
            "topic": user_topic,
            "curriculum": curriculum,
            "current_lesson_index": 0,
            "needs_quiz": False,
        }

    def tutor_node(state: MultiAgentState, config: RunnableConfig):
        """Teach the current lesson from the curriculum."""
        curriculum = state.get("curriculum", [])
        index = state.get("current_lesson_index", 0)
        topic = state.get("topic", "General")
        concept = curriculum[index] if index < len(curriculum) else f"{topic} Overview"

        guide_msg = SystemMessage(
            content=(
                f"You are teaching Lesson {index + 1} of {len(curriculum)}: "
                f"'{concept}' for the topic '{topic}'. "
                f"Teach this lesson clearly and thoroughly. "
                f"At the end, ask the student if they are ready for a quiz."
            )
        )
        temp_state = {**state, "messages": [*state.get("messages", []), guide_msg]}

        res = tutor_agent.invoke(temp_state, config)
        return {
            "messages": res.get("messages", []),
            "needs_quiz": True,
        }

    def examiner_node(state: MultiAgentState, config: RunnableConfig):
        """Generate a quiz question or grade the student's answer."""
        curriculum = state.get("curriculum", [])
        index = state.get("current_lesson_index", 0)
        topic = state.get("topic", "General")
        concept = curriculum[index] if index < len(curriculum) else topic

        # Adaptive difficulty based on student profile
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        profile = db_get_student_profile(thread_id)
        topic_data = profile.get(concept, {})

        difficulty = "medium"
        adaptive_note = ""
        if topic_data.get("incorrect", 0) > topic_data.get("correct", 0):
            difficulty = "easy"
            adaptive_note = (
                f" The student struggled with '{concept}' previously — "
                f"generate an easier question and give extra encouragement."
            )

        guide_msg = SystemMessage(
            content=(
                f"Evaluating Lesson {index + 1}: '{concept}'.{adaptive_note} "
                f"Generate a {difficulty} quiz question on '{concept}', "
                f"or grade the student's answer if they just responded."
            )
        )
        temp_state = {**state, "messages": [*state.get("messages", []), guide_msg]}

        res = examiner_agent.invoke(temp_state, config)

        # Inspect tool results to see if grading occurred
        graded = False
        is_correct = False
        for msg in reversed(res.get("messages", [])):
            if isinstance(msg, ToolMessage) and msg.name == "grade_answer":
                graded = True
                try:
                    is_correct = json.loads(msg.content).get("correct", False)
                except Exception:
                    pass
                break

        if graded and is_correct:
            # Advance to the next lesson
            return {
                "messages": res.get("messages", []),
                "current_lesson_index": index + 1,
                "needs_quiz": False,
            }
        else:
            # Stay on current lesson (wrong answer or quiz just generated)
            return {
                "messages": res.get("messages", []),
                "needs_quiz": True,
            }

    def completion_node(state: MultiAgentState, config: RunnableConfig):
        """Congratulate the student on finishing all lessons."""
        topic = state.get("topic", "your chosen topic")
        curriculum = state.get("curriculum", [])
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"🎉 **Congratulations!** You've completed all {len(curriculum)} lessons "
                        f"on **{topic}**!\n\n"
                        f"You've shown great understanding across every lesson. "
                        f"Feel free to say **hi** to start learning a new topic!"
                    )
                )
            ],
        }

    # ═════════════════════════════════════════
    #  ROUTER  (determines which node runs next)
    # ═════════════════════════════════════════

    def route_next(state: MultiAgentState) -> str:
        messages = state.get("messages", [])
        last_human = next(
            (msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)),
            "",
        )

        # ── 1. Greetings / restarts ALWAYS go to greeter ──
        #    This is checked FIRST so stale state never traps the user.
        if _is_greeting(last_human):
            return "greeter"

        # ── 2. No topic selected → user is providing their topic → plan ──
        if not state.get("topic"):
            return "planner"

        # ── 3. No curriculum created yet → plan ──
        if not state.get("curriculum"):
            return "planner"

        # ── 4. All lessons completed → celebrate ──
        if state.get("current_lesson_index", 0) >= len(state.get("curriculum", [])):
            return "completion"

        # ── 5. Quiz / grading pending → examiner ──
        if state.get("needs_quiz"):
            return "examiner"

        # ── 6. Default: teach the next lesson ──
        return "tutor"

    # ═════════════════════════════════════════
    #  BUILD THE GRAPH
    # ═════════════════════════════════════════

    builder = StateGraph(MultiAgentState)
    builder.add_node("greeter", greeter_node)
    builder.add_node("planner", planner_node)
    builder.add_node("tutor", tutor_node)
    builder.add_node("examiner", examiner_node)
    builder.add_node("completion", completion_node)

    builder.add_conditional_edges(
        START,
        route_next,
        {
            "greeter": "greeter",
            "planner": "planner",
            "tutor": "tutor",
            "examiner": "examiner",
            "completion": "completion",
        },
    )

    # Greeter → END   (wait for student to pick a topic)
    builder.add_edge("greeter", END)
    # Planner → Tutor (auto-chain: plan curriculum then immediately teach lesson 1)
    builder.add_edge("planner", "tutor")
    # Tutor → END     (wait for student to say "quiz me" / respond)
    builder.add_edge("tutor", END)
    # Examiner → END  (wait for next interaction)
    builder.add_edge("examiner", END)
    # Completion → END
    builder.add_edge("completion", END)

    return builder.compile(checkpointer=checkpointer)

