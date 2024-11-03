import asyncio
import aiohttp
from typing import Optional, Tuple, Dict, List
from pocketbase import PocketBase
from bs4 import BeautifulSoup
from csv import reader
from anthropic import Anthropic
import os
from html import unescape
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('euler_importer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
MARKDOWN_TEMPLATE = """# 🎯 Бодлого {problem_id}

{title}

## 📝 Бодлогын тайлбар
{description}

## 🔢 Математик илэрхийлэл
{mathematical_expression}

## 🤔 Алхам алхмаар бодох нь
{steps}


## 🌟 Жишээ
{example}

## 🤖 Алгоритм
$$
\\begin{{align*}}
& \\textbf{{Бодлого:}} \\text{{{title}}} \\\\
& \\textbf{{Оролт:}} \\text{{{input_params}}} \\\\
& \\textbf{{Гаралт:}} \\text{{{output_description}}} \\\\
& \\\\
& \\textbf{{Анхны утгуудыг оноох:}} \\\\
& \\quad \\text{{{initialization}}} \\\\
& \\\\
& \\textbf{{Алгоритм:}} \\\\
& \\quad {main_steps} \\\\
& \\\\
& \\textbf{{return}} \\text{{ {return_value} }}
\\end{{align*}}
$$"""

CODE_TEMPLATE = """
def bodlogo_{n}(n: int) -> int:
\t\"\"\"
\t#Бодлого {n}: {title}
\t
\tArgs:
\t\tn (int): {input_description}
\t
\tReturns:
\t\tint: {output_description}
\t\"\"\"
\t# Анхны утгуудыг оноох
\t{initialization}
\t
\t# Үндсэн алхмууд
\t{main_logic}
\t
\treturn hariu

# Жижиг тоогоор туршиж үзэх
print(f"Жишээ тест: {bodlogo_{n}(10)}")
"""

EXAMPLE_PYTHON_CELL = """# Бодлогын функц
def bodlogo_{problem_id}(n: int) -> int:
\t# Нийлбэрийг хадгалах хувьсагч
\tniilber = 0
\t
\t# Тайлбар бичих
\tfor too in range(1, n + 1):
\t\t# Нөхцөл шалгах
\t\tif too % 3 == 0 or too % 5 == 0:
\t\t\tniilber = niilber + too
\t
\treturn niilber

# Жижиг тоогоор туршиж үзэх
print('10 хүртэлх тооны нийлбэр:', bodlogo_{problem_id}(10))

# Бодлогын хариуг олох
hariu = bodlogo_{problem_id}(1000)
print('Хариу:', hariu)"""

class EulerImporter:
    def __init__(self, site_id: str, anthropic_api_key: str, pb_url: str = "http://127.0.0.1:8090"):
        self.pb = PocketBase(pb_url)
        self.site_id = site_id
        logger.info(f"Initializing EulerImporter with site_id: {site_id}, pb_url: {pb_url}")
        
        start_time = time.time()
        self.answers = self.load_answers()
        logger.info(f"Loaded {len(self.answers)} answers in {time.time() - start_time:.2f} seconds")
        
        self.anthropic = Anthropic(api_key=anthropic_api_key)
        self.session = None
        self.semaphore = asyncio.Semaphore(5)
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        
        # Track API calls and timing
        self.api_calls = {
            'anthropic': 0,
            'projecteuler': 0,
            'pocketbase': 0
        }
        self.timing_stats = {
            'translation': [],
            'editorial': [],
            'fetch': []
        }
    
    def load_answers(self) -> Dict[int, str]:
        """Load answers from CSV file"""
        try:
            with open('hello.csv') as file:
                csv_reader = reader(file)
                answers = {i: ans[0] for i, ans in enumerate(csv_reader, 1) if ans}
            logger.debug(f"Loaded answers: {list(answers.keys())}")
            return answers
        except Exception as e:
            logger.error(f"Error loading answers: {str(e)}")
            raise

    async def validate_python_code(self, code: str) -> bool:
        """Validate Python code in a separate thread"""
        try:
            def compile_code():
                compile(code, '<string>', 'exec')
            await asyncio.get_event_loop().run_in_executor(self.thread_pool, compile_code)
            return True
        except Exception as e:
            logger.error(f"Code validation error: {str(e)}\nCode:\n{code}")
            return False

    async def translate_to_mongolian(self, text: str) -> str:
        """Translate text to Mongolian using Claude"""
        start_time = time.time()
        async with self.semaphore:
            try:
                logger.debug("Starting translation")
                message = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    temperature=0,
                    system="You are a professional translator specializing in mathematical and problem-solving content. Translate the given text from English to Mongolian, maintaining all mathematical notation and formatting. Preserve any HTML tags in their original form.",
                    messages=[{
                        "role": "user",
                        "content": f"Translate this text to Mongolian. Keep all mathematical notations, numbers, and HTML formatting intact:\n\n{text}"
                    }]
                )
                self.api_calls['anthropic'] += 1
                duration = time.time() - start_time
                self.timing_stats['translation'].append(duration)
                logger.debug(f"Translation completed in {duration:.2f} seconds")
                return message.content[0].text
            except Exception as e:
                logger.error(f"Translation error: {str(e)}\nText:\n{text[:200]}...")
                return ""

    async def fetch_problem_title(self, session: aiohttp.ClientSession, problem_id: int) -> Optional[str]:
        """Fetch just the title from the full problem page"""
        async with self.semaphore:
            url = f'https://projecteuler.net/problem={problem_id}'
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        title = soup.find('h2')
                        self.api_calls['projecteuler'] += 1
                        return title.text.strip() if title else None
            except Exception as e:
                logger.error(f"Error fetching title for problem {problem_id}: {str(e)}")
                return None

    async def fetch_problem_content(self, session: aiohttp.ClientSession, problem_id: int) -> Optional[str]:
        """Fetch just the content from the minimal version"""
        async with self.semaphore:
            url = f'https://projecteuler.net/minimal={problem_id}'
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        self.api_calls['projecteuler'] += 1
                        return await response.text()
            except Exception as e:
                logger.error(f"Error fetching content for problem {problem_id}: {str(e)}")
                return None

    async def generate_editorial(self, problem_id: int, title: str, content: str, answer: str) -> str:
        """Generate editorial content using Claude"""
        start_time = time.time()
        async with self.semaphore:
            try:
                logger.debug(f"Generating editorial for problem {problem_id}")
                example_code = EXAMPLE_PYTHON_CELL.format(problem_id=problem_id)
                
                prompt = f"""Create a tutorial notebook in Mongolian for this Project Euler problem. Use LaTeX for mathematical expressions (enclosed in $ signs) and proper markdown formatting.

Problem {problem_id}: {title}
Content: {content}
Answer: {answer}

Follow these format requirements exactly:

Key requirements:
1. Use LaTeX for all mathematical expressions ($...$ for inline, $$...$$ for blocks)
2. Code should show only the core concept
3. Use proper markdown formatting including code blocks with ```
4. Include visual representations of the problem where possible
5. Explain the mathematical concept clearly

1. Markdown cells should be engaging and visual:
- Use emojis appropriately
- Include clear step-by-step explanations
- Add visual examples where possible
- Format mathematical expressions properly
- Include helpful tips and insights

2. Python code must follow these rules exactly:
- It should be easy to understand highschool students and beginners
- Use tabs and (\\t) for indentation
- Every function should start with 'bodlogo_'
- One operation per line
- Clear comment for each section

3. **For the pseudocode in the "## 🤖 Алгоритм" section:**
- **DO NOT USE UNDERSCORE IN LATEX**
- **Always use english letters**
- **Enclose the pseudocode within LaTeX environments as per the template**
- **Use LaTeX formatting for any mathematical symbols or expressions within the pseudocode**
- **Ensure the pseudocode is clear and properly formatted**

Here's the required Markdown structure:
{MARKDOWN_TEMPLATE}

Here's the required Python code structure:
{CODE_TEMPLATE}

The final JSON must have this exact format:
{{
    "cells": [
        {{
            "id": 1,
            "type": "markdown",
            "content": "[Markdown following the template above]",
            "metadata": {{}}
        }},
        {{
            "id": 2,
            "type": "code",
            "content": "[Python code following the template above]",
            "metadata": {{"language": "python"}},
            "output": "[Гаралт]"
        }},
        {{
            "id": 3,
            "type": "markdown",
            "content": "## 💡 Нэмэлт тайлбар\\n\\n1. [Ашигласан аргын тайлбар]\\n2. [Кодны гол хэсгүүдийн тайлбар]\\n3. [Цаашид юу анхаарах тухай]",
            "metadata": {{}}
        }}
    ],
    "metadata": {{
        "kernelspec": {{
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }},
        "language_info": {{
            "name": "python",
            "version": "3.9.0"
        }}
    }},
    "nbformat": 4,
    "nbformat_minor": 5
}}"""

                message = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4096,
                    temperature=0,
                    system="You are a Python teacher creating content for Mongolian high school students. Write clear explanations and simple, well-structured code that students can easily understand. Use Mongolian comments. Show example runs with small numbers first.",
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )

                self.api_calls['anthropic'] += 1
                content = message.content[0].text
                
                try:
                    parsed = json.loads(content)
                    logger.debug("Successfully parsed editorial JSON")
                    
                    for cell in parsed.get('cells', []):
                        if cell.get('type') == 'code':
                            code_content = cell.get('content', '')
                            logger.debug(f"Validating Python code for problem {problem_id}")
                            is_valid = await self.validate_python_code(code_content)
                            if not is_valid:
                                logger.error(f"Invalid Python code in problem {problem_id}")
                                return ""
                            logger.debug("Code validation successful")
                    
                    duration = time.time() - start_time
                    self.timing_stats['editorial'].append(duration)
                    logger.debug(f"Editorial generation completed in {duration:.2f} seconds")
                    return content
                    
                except json.JSONDecodeError:
                    logger.warning("JSON parse failed, attempting to extract JSON portion")
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start >= 0 and end > start:
                        return content[start:end]
                    logger.error("Failed to extract valid JSON")
                    return ""
                    
            except Exception as e:
                logger.error(f"Editorial generation error: {str(e)}")
                return ""


    async def fetch_and_process_problem(self, problem_id: int) -> Optional[Tuple[str, str, str, str, str]]:
        """Fetch and process a single problem"""
        start_time = time.time()
        if problem_id not in self.answers:
            logger.debug(f"Skipping problem {problem_id} - no answer available")
            return None
            
        try:
            logger.info(f"Processing problem {problem_id}")
            title_task = self.fetch_problem_title(self.session, problem_id)
            content_task = self.fetch_problem_content(self.session, problem_id)
            
            title, content = await asyncio.gather(title_task, content_task)
            logger.debug(f"Fetched problem {problem_id} - Title: {title[:30]}...")
            
            if title and content:
                content = unescape(content)
                
                translation_task = self.translate_to_mongolian(content)
                editorial_task = self.generate_editorial(
                    problem_id, 
                    title, 
                    content, 
                    self.answers[problem_id]
                )
                
                content_mn, editorial = await asyncio.gather(translation_task, editorial_task)
                
                duration = time.time() - start_time
                self.timing_stats['fetch'].append(duration)
                logger.info(f"Completed problem {problem_id} in {duration:.2f} seconds")
                
                return (title, content, '', content_mn, editorial)
                
        except Exception as e:
            logger.error(f"Error processing problem {problem_id}: {str(e)}")
        
        return None

    async def insert_problem(self, problem_id: int, problem_data: Tuple[str, str, str, str, str]) -> None:
        """Insert a problem into the database"""
        title, content, title_mn, content_mn, editorial = problem_data
        try:
            self.pb.collection('problems').create({
                "description_en": content,
                "description_mn": content_mn,
                "answer": self.answers.get(problem_id, ""),
                "site": self.site_id,
                "title_en": title,
                "title_mn": title_mn,
                "order": problem_id,
                "explanation_en": {},
                "explanation_mn": editorial
            })
            self.api_calls['pocketbase'] += 1
            logger.info(f"Successfully inserted problem {problem_id}: {title}")
        except Exception as e:
            logger.error(f"Failed to insert problem {problem_id}: {str(e)}")

    async def import_problems(self, start: int = 1, end: int = 10):
        """Import a range of Project Euler problems concurrently"""
        total_start_time = time.time()
        logger.info(f"Starting import of problems {start} to {end}")
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            batch_size = 5
            for i in range(start, end + 1, batch_size):
                batch_start_time = time.time()
                batch_end = min(i + batch_size, end + 1)
                logger.info(f"Processing batch: problems {i} to {batch_end-1}")
                
                tasks = [self.fetch_and_process_problem(j) for j in range(i, batch_end)]
                results = await asyncio.gather(*tasks)
                
                successful = 0
                for j, result in enumerate(results, i):
                    if result:
                        await self.insert_problem(j, result)
                        successful += 1
                
                batch_duration = time.time() - batch_start_time
                logger.info(f"Batch completed in {batch_duration:.2f} seconds. {successful}/{batch_end-i} problems processed successfully")
                
                # Small delay between batches to avoid rate limiting
                await asyncio.sleep(1)
        
        total_duration = time.time() - total_start_time
        self.log_statistics(total_duration, start, end)

    def log_statistics(self, total_duration: float, start: int, end: int):
        """Log performance statistics"""
        problems_processed = end - start + 1
        logger.info("\n=== Performance Statistics ===")
        logger.info(f"Total time: {total_duration:.2f} seconds")
        logger.info(f"Average time per problem: {total_duration/problems_processed:.2f} seconds")
        logger.info("\nAPI Calls:")
        for api, count in self.api_calls.items():
            logger.info(f"- {api}: {count} calls")
        logger.info("\nTiming Statistics:")
        for operation, times in self.timing_stats.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                logger.info(f"- {operation}:")
                logger.info(f"  * Average: {avg_time:.2f} seconds")
                logger.info(f"  * Maximum: {max_time:.2f} seconds")

async def main():
    site_id = "aljwrgk9b2xtw8j"
    anthropic_api_key = "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"
    
    try:
        importer = EulerImporter(site_id, anthropic_api_key, 'https://api.edukit.mn')
        await importer.import_problems(11, 20)
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())