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
    topic: str
    curriculum: list[str]
    current_lesson_index: int
    needs_quiz: bool

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

    # 1. Greeting Agent: welcomes student, asks what topic they want to learn
    greeter_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=SystemMessage(content="""You are an encouraging, expert AI tutor.
The student has just greeted you or started a new session.
Greet the student warmly and ask them what subject or topic they would like to learn today (for example: AI Agents, Python, Biology, Operating Systems, Math, etc.).
Do not teach any lesson or ask quiz questions yet. Simply welcome them and ask for their desired learning topic."""),
        middleware=middleware_common
    )

    # 2. Planner Agent: creates a structured 3-part curriculum for the chosen topic
    planner_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=SystemMessage(content="""You are a master curriculum planner.
The student has chosen a topic to learn.
Acknowledge their chosen topic enthusiastically and introduce a clear, structured 3-lesson curriculum for them.
Format your response clearly:
"Awesome! Let's learn **[Topic]**. Here is our 3-part learning curriculum:
1. [Lesson 1 Title]: Brief description
2. [Lesson 2 Title]: Brief description
3. [Lesson 3 Title]: Brief description

Let's begin with Lesson 1!"
"""),
        middleware=middleware_common
    )

    # 3. Tutor Agent: teaches/explains the current lesson
    from tools import explain_concept, retrieve_reference
    tutor_agent = create_agent(
        model=model,
        tools=[explain_concept, retrieve_reference],
        system_prompt=SystemMessage(content="""You are an expert tutor.
Your job is to clearly explain and teach the current lesson concept from the curriculum.
Use `explain_concept` or `retrieve_reference` to ground your explanation.
Explain the concepts engagingly with intuitive analogies and real-world examples.
At the very end of your explanation, say: "Ready to test your understanding with a quick quiz?" and encourage the student."""),
        middleware=middleware_common
    )

    # 4. Examiner Agent: quizzes, grades, updates scores
    from tools import generate_quiz_question, grade_answer
    middleware_examiner = list(middleware_common)
    middleware_examiner.append(AnswerGuardrailMiddleware(enabled=guardrail_enabled))
    middleware_examiner.append(GradingLoggingMiddleware(enabled=grading_log_enabled))
    middleware_examiner.append(HITLInterruptMiddleware(enabled=hitl_enabled))
    
    examiner_agent = create_agent(
        model=model,
        tools=[generate_quiz_question, grade_answer],
        system_prompt=SystemMessage(content="""You are an examiner agent.
Your job is to test the student's understanding of the lesson.
Call `generate_quiz_question` to ask a high-quality multiple choice question on the lesson.
When the student provides their answer (e.g. 'I choose answer option A' or 'A'), invoke `grade_answer` to evaluate it accurately.
If correct, congratulate them on mastering this lesson!
If incorrect, provide helpful encouragement and explain the concept clearly."""),
        middleware=middleware_examiner
    )

    # Node: Greeter
    def greeter_node(state: MultiAgentState, config: RunnableConfig):
        res = greeter_agent.invoke(state, config)
        return {
            "messages": res.get("messages", []),
            "topic": "",
            "curriculum": [],
            "current_lesson_index": 0,
            "needs_quiz": False
        }

    # Node: Planner
    def planner_node(state: MultiAgentState, config: RunnableConfig):
        messages = state.get("messages", [])
        user_msg = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), "AI Agents")
        
        # Invoke planner agent
        res = planner_agent.invoke(state, config)
        
        # Extract structured 3 topics
        prompt = (
            f"From this curriculum text, extract a JSON list of the 3 lesson titles:\n"
            f"\"\"\"{res.get('messages', [])[-1].content}\"\"\"\n"
            f"Return ONLY a valid JSON list of 3 strings (e.g. [\"Lesson 1\", \"Lesson 2\", \"Lesson 3\"])."
        )
        try:
            parse_res = model.invoke(prompt)
            content = parse_res.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            curriculum = json.loads(content.strip())
            if not isinstance(curriculum, list) or len(curriculum) < 2:
                curriculum = [f"{user_msg} Fundamentals", f"{user_msg} Core Architecture", f"{user_msg} Advanced Applications"]
        except Exception:
            curriculum = [f"{user_msg} Fundamentals", f"{user_msg} Core Architecture", f"{user_msg} Advanced Applications"]
            
        return {
            "messages": res.get("messages", []),
            "topic": user_msg,
            "curriculum": curriculum,
            "current_lesson_index": 0,
            "needs_quiz": False
        }

    # Node: Tutor (Teaches current lesson)
    def tutor_node(state: MultiAgentState, config: RunnableConfig):
        curriculum = state.get("curriculum", [])
        index = state.get("current_lesson_index", 0)
        topic = state.get("topic", "General")
        concept = curriculum[index] if index < len(curriculum) else f"{topic} Overview"
        
        guide_msg = SystemMessage(
            content=f"You are teaching Lesson {index+1} of {len(curriculum)}: '{concept}' for the topic '{topic}'. "
                    f"Teach this lesson clearly and thoroughly. At the end, ask the student if they are ready for a quiz."
        )
        temp_state = {**state, "messages": [*state.get("messages", []), guide_msg]}
        
        res = tutor_agent.invoke(temp_state, config)
        return {
            "messages": res.get("messages", []),
            "needs_quiz": True
        }

    # Node: Examiner (Generates quiz or grades answer)
    def examiner_node(state: MultiAgentState, config: RunnableConfig):
        curriculum = state.get("curriculum", [])
        index = state.get("current_lesson_index", 0)
        topic = state.get("topic", "General")
        concept = curriculum[index] if index < len(curriculum) else topic
        
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        profile = db_get_student_profile(thread_id)
        topic_profile = profile.get(concept, {})
        
        difficulty = "medium"
        adaptive_msg = ""
        if topic_profile.get("incorrect", 0) > topic_profile.get("correct", 0):
            difficulty = "easy"
            adaptive_msg = f"NOTE: The student struggled with '{concept}' previously. Generate an easy question and give extra encouragement."
            
        guide_msg = SystemMessage(
            content=f"You are evaluating Lesson {index+1}: '{concept}'. "
                    f"If a question needs to be asked, generate a {difficulty} question on '{concept}'. "
                    f"If the student just submitted an answer, grade it using grade_answer. {adaptive_msg}"
        )
        temp_state = {**state, "messages": [*state.get("messages", []), guide_msg]}
        
        res = examiner_agent.invoke(temp_state, config)
        
        # Check if an answer was graded
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
                
        if graded:
            if is_correct:
                next_index = index + 1
                return {
                    "messages": res.get("messages", []),
                    "current_lesson_index": next_index,
                    "needs_quiz": False
                }
            else:
                # Keep on current lesson until mastered or retried
                return {
                    "messages": res.get("messages", []),
                    "needs_quiz": True
                }
        else:
            return {
                "messages": res.get("messages", []),
                "needs_quiz": True
            }

    # Router logic
    def route_next(state: MultiAgentState):
        messages = state.get("messages", [])
        last_human = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), "").strip()
        last_human_lower = last_human.lower()
        
        # Check for simple greetings
        greetings = ["hi", "hey", "hello", "good morning", "good evening", "howdy", "start", "restart"]
        is_simple_greeting = last_human_lower in greetings or any(last_human_lower.startswith(g + " ") for g in ["hi", "hey", "hello"]) and len(last_human.split()) <= 3
        
        # If no topic has been established yet
        if not state.get("topic"):
            if is_simple_greeting:
                return "greeter"
            else:
                # User provided their desired topic (e.g. "I want to learn AI Agents" or "Python")
                return "planner"
                
        # If curriculum is empty or not yet initialized
        if not state.get("curriculum"):
            return "planner"
            
        # Check if all lessons are completed
        if state.get("current_lesson_index", 0) >= len(state.get("curriculum", [])):
            return "end"
            
        # Check if the student needs a quiz or just answered a quiz
        if state.get("needs_quiz"):
            return "examiner"
            
        # Otherwise, teach the next lesson
        return "tutor"

    # Construct the Parent Graph
    builder = StateGraph(MultiAgentState)
    builder.add_node("greeter", greeter_node)
    builder.add_node("planner", planner_node)
    builder.add_node("tutor", tutor_node)
    builder.add_node("examiner", examiner_node)
    
    builder.add_conditional_edges(
        START,
        route_next,
        {
            "greeter": "greeter",
            "planner": "planner",
            "tutor": "tutor",
            "examiner": "examiner",
            "end": END
        }
    )
    
    builder.add_edge("greeter", END)
    builder.add_edge("planner", END)
    builder.add_edge("tutor", END)
    builder.add_edge("examiner", END)
    
    return builder.compile(checkpointer=checkpointer)
