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

class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    curriculum: list[str]
    current_lesson_index: int
    needs_quiz: bool
    topic: str

def build_agent(
    checkpointer,
    model_retry_enabled=True,
    tool_retry_enabled=True,
    limit_enabled=True,
    guardrail_enabled=True,
    grading_log_enabled=True,
    hitl_enabled=True
):
    # Setup sub-agent middlewares
    middleware_common = []
    if model_retry_enabled:
        middleware_common.append(ModelRetryMiddleware(max_retries=3))
    if tool_retry_enabled:
        middleware_common.append(ToolRetryMiddleware(max_retries=3))
    if limit_enabled:
        middleware_common.append(ModelCallLimitMiddleware(run_limit=10, exit_behavior="error"))

    # Planner Agent: creates a curriculum
    planner_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=SystemMessage(content="""You are a curriculum planner.
Your job is to take the student's topic and create a numbered curriculum of exactly 3 topics.
Output the curriculum clearly to the student, e.g.:
'Here is our curriculum:
1. topic A
2. topic B
3. topic C'"""),
        middleware=middleware_common
    )

    # Tutor Agent: teaches/explains the current lesson
    from tools import explain_concept, retrieve_reference
    tutor_agent = create_agent(
        model=model,
        tools=[explain_concept, retrieve_reference],
        system_prompt=SystemMessage(content="""You are an expert tutor.
Your job is to explain the current concept of the curriculum.
Call explain_concept to teach the topic. Ground your explanation in reference materials.
At the end of your response, offer a quiz question by saying 'Ready for a quick quiz?'"""),
        middleware=middleware_common
    )

    # Examiner Agent: quizzes, grades, updates scores
    from tools import generate_quiz_question, grade_answer
    middleware_examiner = list(middleware_common)
    middleware_examiner.append(AnswerGuardrailMiddleware(enabled=guardrail_enabled))
    middleware_examiner.append(GradingLoggingMiddleware(enabled=grading_log_enabled))
    middleware_examiner.append(HITLInterruptMiddleware(enabled=hitl_enabled))
    
    examiner_agent = create_agent(
        model=model,
        tools=[generate_quiz_question, grade_answer],
        system_prompt=SystemMessage(content="""You are an examiner agent.
Your job is to test the student's understanding.
Call generate_quiz_question to ask a quiz question.
When the student responds, call grade_answer to grade it.
If they are correct, praise them. If they are incorrect, explain why."""),
        middleware=middleware_examiner
    )

    # 1. Planner Node
    def planner_node(state: MultiAgentState, config: RunnableConfig):
        messages = state.get("messages", [])
        user_msg = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), "Python loops")
        
        res = planner_agent.invoke(state, config)
        
        curriculum = []
        prompt = (
            f"Extract a JSON list of the 3 curriculum topics from this curriculum text:\n"
            f"\"\"\"{res.get('messages', [])[-1].content}\"\"\"\n"
            f"Only return a valid JSON list of strings (e.g. [\"A\", \"B\", \"C\"]), nothing else."
        )
        parse_res = model.invoke(prompt)
        try:
            content = parse_res.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            curriculum = json.loads(content.strip())
        except Exception:
            curriculum = ["Basics", "Intermediate", "Advanced"]
            
        return {
            "messages": res.get("messages", []),
            "curriculum": curriculum,
            "current_lesson_index": 0,
            "needs_quiz": False,
            "topic": user_msg
        }

    # 2. Tutor Node
    def tutor_node(state: MultiAgentState, config: RunnableConfig):
        curriculum = state.get("curriculum", [])
        index = state.get("current_lesson_index", 0)
        concept = curriculum[index] if index < len(curriculum) else "General"
        
        guide_msg = SystemMessage(content=f"You are teaching Lesson {index+1}: '{concept}'. Explain this concept now.")
        temp_state = {**state, "messages": [*state.get("messages", []), guide_msg]}
        
        res = tutor_agent.invoke(temp_state, config)
        return {
            "messages": res.get("messages", []),
            "needs_quiz": True
        }

    # 3. Examiner Node
    def examiner_node(state: MultiAgentState, config: RunnableConfig):
        curriculum = state.get("curriculum", [])
        index = state.get("current_lesson_index", 0)
        concept = curriculum[index] if index < len(curriculum) else "General"
        
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        profile = db_get_student_profile(thread_id)
        topic_profile = profile.get(concept, {})
        
        difficulty = "medium"
        adaptive_msg = ""
        if topic_profile.get("incorrect", 0) > topic_profile.get("correct", 0):
            difficulty = "easy"
            adaptive_msg = f"GUIDELINE: The student struggled with '{concept}' previously. You MUST ask an easy question and mention: 'You struggled with this topic, so let's try an easier question to practice!'"
            
        guide_msg = SystemMessage(content=f"Test the student's understanding of '{concept}'. Ask a {difficulty} difficulty question. {adaptive_msg}")
        temp_state = {**state, "messages": [*state.get("messages", []), guide_msg]}
        
        res = examiner_agent.invoke(temp_state, config)
        
        is_correct = False
        for msg in reversed(res.get("messages", [])):
            if isinstance(msg, ToolMessage) and msg.name == "grade_answer":
                try:
                    is_correct = json.loads(msg.content).get("correct", False)
                except Exception:
                    pass
                break
                
        if is_correct:
            return {
                "messages": res.get("messages", []),
                "current_lesson_index": index + 1,
                "needs_quiz": False
            }
        else:
            return {
                "messages": res.get("messages", []),
                "needs_quiz": True
            }

    # Routing logic
    def route_next(state: MultiAgentState):
        if not state.get("curriculum"):
            return "planner"
        if state.get("current_lesson_index", 0) >= len(state.get("curriculum", [])):
            return "end"
        if state.get("needs_quiz"):
            return "examiner"
        return "tutor"

    # Construct the Parent Graph
    builder = StateGraph(MultiAgentState)
    builder.add_node("planner", planner_node)
    builder.add_node("tutor", tutor_node)
    builder.add_node("examiner", examiner_node)
    
    builder.add_conditional_edges(
        START,
        route_next,
        {
            "planner": "planner",
            "tutor": "tutor",
            "examiner": "examiner",
            "end": END
        }
    )
    
    builder.add_edge("planner", END)
    builder.add_edge("tutor", END)
    builder.add_edge("examiner", END)
    
    return builder.compile(checkpointer=checkpointer)
