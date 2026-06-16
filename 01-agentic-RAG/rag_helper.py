
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

from pydantic_ai import Agent


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

    def search(self, query, num_results=5):
        # Fixed: Removed the broken set definitions completely
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
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]
        agent = Agent(
        model=self.llm_client,
        system_prompt=prompt,
        )
        result = await agent.run()
        # Fixed: Returning usage metadata so you can answer Q3
        return result.output_text, result.usage

    async def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer, usage = await self.llm(prompt)
        print(f"Token usage info: {usage}") 
        return answer , usage