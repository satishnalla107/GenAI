import os
from langchain_groq  import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

## Get LLM
def get_llm(model_name:str="openai/gpt-oss-20b",temperature:float = 0.5):
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=model_name,temperature=temperature,api_key=api_key)
    return llm


### Research Agent
RESEARCHER_PROMPT=ChatPromptTemplate.from_messages([
    {"role":"System","content":"""
     "You are a Research Agent. Given a blog topic and target audience, produce a clear, "
        "structured research outline. Include: in"
        "1. 5-7 key points the blog should covenin"
        "2. Important facts, stats, or examples for each pointin"
        "3. Suggested angle or hookin"
        "Be concise. Use bullet points. Do NOT write the full blog yet."
        """},
    {"role":"user","content":"Topic:{topic} , Audience:{audience}, {revision_hints}, Now write research outline now. "}])


def researcher_agent (llm: ChatGroq, topic: str, audience: str, feedback:str ="")-> str:
    revision_hints = f"The human provided this feedback on your previous research - please address it: {feedback}."
    if not feedback:
         revision_hints = "This is your first attempt."
    chain = RESEARCHER_PROMPT | llm
    result = chain.invoke({
        "topic": topic,
        "audience": audience,
        "revision_hints":revision_hints })
    return result.content



### Writer Prompt
WRITER_PROMPT=ChatPromptTemplate.from_messages([
    {"role":"System", "content": """
        You are a Blog Writer Agent.

        Using the research notes provided, write a complete, engaging blog post.

        Rules:
        - Length: 500 to 800 words
        - Structure:
        - Catchy title
        - Introduction with a strong hook
        - 3 to 5 sections with H2 headings
        - Conclusion
        - Tone: Clear, friendly, and suited to the target audience
        - Use Markdown formatting
        - Do NOT add a 'Word Count' line at the end.
        """},
    {"role":"user","content":"""
         Topic:{topic} ,
         Audience:{audience}, 
         Research Notes :{research} {revision_hints},
         Write the full blog post now . """}])


def writer_agent (llm: ChatGroq, topic: str, audience: str, research :str="",feedback:str ="")-> str:
    revision_hints = f"The human provided this feedback on your draft and asked these changes: {feedback} and please apply these changes during writing the blog."
    if not feedback:
         revision_hints = "This is your first attempt."
    chain = WRITER_PROMPT | llm
    result = chain.invoke({
        "topic": topic,
        "audience": audience,
        "research":research,
        "revision_hints":revision_hints })
    return result.content


### Editor Agent

EDITOR_PROMPT=ChatPromptTemplate.from_messages([
    {"role":"System",     "content": """
        You are an Editor Agent, the final quality gate before publishing.

        Your task is to review the draft blog post and produce the final polished version.

        Guidelines:
        - Correct grammar, spelling, punctuation, and formatting errors.
        - Improve clarity by rewriting awkward or unclear sentences.
        - Tighten wordy or repetitive sentences while preserving the original meaning.
        - Enhance the flow and transitions between sections.
        - Make the title and introduction more engaging if needed.
        - Maintain a consistent tone that is appropriate for the target audience.
        - Preserve the original structure, headings, and Markdown formatting.
        - Do not introduce new facts or remove important information from the draft.

        Output only the final polished blog post. Do not include explanations, comments, or any additional text.
        """  },
    {"role":"user","content":"""
         Topic:{topic} ,
         Draft :{draft},
         Return the published blog. 
     """}])


def editor_agent (llm: ChatGroq, topic: str, draft:str ="")-> str:
    
    chain = EDITOR_PROMPT | llm
    result = chain.invoke({
        "topic": topic,
        "draft":draft,
         })
    return result.content
