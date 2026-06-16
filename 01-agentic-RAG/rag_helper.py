

from pydantic_ai import Agent

INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='gpt-5.4-mini'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model
        
        # Instantiating the Agent once in __init__ is cleaner and more efficient
        self.agent = Agent(
            model=self.llm_client,
            system_prompt=self.instructions
        )

    def search(self, query, num_results=5):
        return self.index.search(
            query=query,
            num_results=num_results
        )

    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(doc['content'])
            lines.append('')
        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    async def llm(self, prompt):
        result = await self.agent.run(prompt)
        return result.output, result.usage

    async def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer, usage = await self.llm(prompt)
        
        # This will now cleanly print out your prompt token counts for Q3!
        print(f"Token usage info: {usage}") 
        return answer, usage