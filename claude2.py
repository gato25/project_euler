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

# At the top of the file
import logging
import sys
import codecs

# Set up logging
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Set up logging with reasonable defaults
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('euler_importer.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create a debug file handler for detailed debugging
debug_handler = logging.FileHandler('euler_importer_debug.log', encoding='utf-8')
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(debug_handler)

MARKDOWN_TEMPLATE_EN = """# 🎯 Problem {problem_id}

{title}

## 📝 Problem Description
{description}

## 🔢 Mathematical Expression
{mathematical_expression}

## 🤔 Step by Step Solution
{steps}

## 🌟 Example
{example}

## 🤖 Algorithm
$$
\\begin{{align*}}
& \\textbf{{Problem:}} \\text{{{title}}} \\\\
& \\textbf{{Input:}} \\text{{{input_params}}} \\\\
& \\textbf{{Output:}} \\text{{{output_description}}} \\\\
& \\\\
& \\textbf{{Initialize:}} \\\\
& \\quad \\text{{{initialization}}} \\\\
& \\\\
& \\textbf{{Algorithm:}} \\\\
& \\quad {main_steps} \\\\
& \\\\
& \\textbf{{return}} \\text{{ {return_value} }}
\\end{{align*}}
$$

## 💡 Additional Notes
1. {concept_notes}
2. {implementation_notes}
3. {optimization_tips}
"""

MARKDOWN_TEMPLATE_MN = """# 🎯 Бодлого {problem_id}

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
$$

## 💡 Нэмэлт тайлбар
1. {concept_notes}
2. {implementation_notes}
3. {optimization_tips}
"""

CODE_TEMPLATE_EN = """
def problem_{n}(n: int) -> int:
\t\"\"\"
\t#Problem {n}: {title}
\t  
\tArgs:
\t\tn (int): {input_description}
\t
\tReturns:
\t\tint: {output_description}
\t\"\"\"
\t# Initialize variables
\t{initialization}
\t  
\t# Main steps
\t{main_logic}
\t    
\treturn result
"""

CODE_TEMPLATE_MN = """
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
        
        self.api_calls = {
            'anthropic': 0,
            'projecteuler': 0,
            'pocketbase': 0
        }
        self.timing_stats = {
            'translation': [],
            'explanation': [],
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

    async def generate_explanation(self, problem_id: int, title: str, content: str, answer: str, language: str = "en") -> str:
        """Generate explanation content in specified language using Claude"""
        start_time = time.time()
        async with self.semaphore:
            try:
                prompt_language = 'Mongolian' if language == 'mn' else 'English'
                template = MARKDOWN_TEMPLATE_MN if language == "mn" else MARKDOWN_TEMPLATE_EN
                code_template = CODE_TEMPLATE_MN if language == "mn" else CODE_TEMPLATE_EN
                logger.info(f"Generating {language} explanation for problem {problem_id}")
                
                system_prompt = f"""You are a mathematics professor creating tutorial content for high school students in {prompt_language}.
                Write clear explanations and properly formatted mathematical content.
                Return the response in the exact JSON format specified in the prompt.
                All content must be in {prompt_language}.
                Use LaTeX for mathematical expressions.
                Each cell's content must exactly follow the templates provided."""
                
                prompt = f"""Create a tutorial notebook for Problem {problem_id}.

        Content to explain:
        Title: {title}
        Problem: {content}
        Answer: {answer}

        The first markdown cell must follow this exact template:
        {template}

        The code cell must follow this exact template:
        {code_template}

        Return a JSON object with exactly this structure, following these requirements:

        1. cell 1 (markdown):
        - Must use the provided markdown template, replacing all placeholders
        - All mathematical expressions in LaTeX ($...$ for inline, $$....$$ for blocks)
        - Clear step-by-step explanations
        - Include both conceptual and visual examples
        - All text in {prompt_language}

        2. cell 2 (code):
        - Must use the provided code template
        - Replace all placeholders in the template
        - Include clear comments in {prompt_language}
        - Show example runs with small numbers
        - Must return correct answer for test cases

        3. cell 3 (markdown):
        - Title: "## 💡 {'Нэмэлт тайлбар' if language == 'mn' else 'Additional Notes'}"
        - Three bullet points explaining:
            1. Key mathematical concepts used
            2. Important implementation details
            3. Optimization possibilities

        The JSON must exactly match this structure:
        {{
        "cells": [
            {{
                "id": 1,
                "type": "markdown",
                "content": "[Markdown cell using the provided template]",
                "metadata": {{}}
            }},
            {{
                "id": 2,
                "type": "code",
                "content": "[Code cell using the provided template]",
                "metadata": {{"language": "python"}},
                "output": "[Example test outputs]"
            }},
            {{
                "id": 3,
                "type": "markdown",
                "content": "[Additional notes cell]",
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
        }}

        Fill in all placeholders in both templates with appropriate content in {prompt_language}."""

                message = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    temperature=0,
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )

                self.api_calls['anthropic'] += 1
                duration = time.time() - start_time
                self.timing_stats['explanation'].append(duration)
                
                 # Get the response text
                response_text = message.content[0].text
                
                # Log the raw response
                with open(f'response_{problem_id}_{language}.txt', 'w', encoding='utf-8') as f:
                    f.write(response_text)
                logger.info(f"Saved raw response to response_{problem_id}_{language}.txt")
                
                # Try to find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start == -1 or json_end <= json_start:
                    logger.error(f"No JSON found in response for problem {problem_id}")
                    return ""
                
                json_str = response_text[json_start:json_end]
                
                # Try to clean up common JSON issues
                json_str = json_str.replace('\n', '\\n')  # Fix newline handling
                json_str = json_str.replace('```json\n', '').replace('\n```', '')  # Remove code blocks if present
                
                # Log the extracted JSON
                with open(f'json_{problem_id}_{language}.txt', 'w', encoding='utf-8') as f:
                    f.write(json_str)
                logger.info(f"Saved extracted JSON to json_{problem_id}_{language}.txt")
                
                try:
                    # Try to parse the JSON with detailed error handling
                    try:
                        response = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # Get the problematic line
                        lines = json_str.split('\n')
                        error_line_no = e.lineno - 1
                        error_line = lines[error_line_no] if error_line_no < len(lines) else "Line not found"
                        context_start = max(0, error_line_no - 2)
                        context_end = min(len(lines), error_line_no + 3)
                        context_lines = lines[context_start:context_end]
                        
                        logger.error(f"JSON parse error for problem {problem_id}:")
                        logger.error(f"Error: {str(e)}")
                        logger.error(f"Error location: line {e.lineno}, column {e.colno}")
                        logger.error("Context:")
                        for i, line in enumerate(context_lines, start=context_start):
                            logger.error(f"{i+1}: {line}")
                            if i == error_line_no:
                                logger.error(" " * (e.colno + len(str(i+1)) + 2) + "^")
                        return ""
                    
                    # Log successful parsing
                    logger.info(f"Successfully parsed JSON for problem {problem_id} ({language})")
                    
                    # Validate structure
                    required_keys = ["cells", "metadata", "nbformat", "nbformat_minor"]
                    missing_keys = [key for key in required_keys if key not in response]
                    if missing_keys:
                        logger.error(f"Missing required keys in response: {missing_keys}")
                        return ""
                    
                    # Log key validation
                    logger.info(f"JSON structure validation passed for problem {problem_id}")
                    
                    return json.dumps(response, ensure_ascii=False, indent=2)
                    
                except Exception as e:
                    logger.error(f"Error processing response for problem {problem_id}: {str(e)}")
                    return ""

            except Exception as e:
                logger.error(f"Explanation generation error: {str(e)}")
                return ""
            
    async def fetch_and_process_problem(self, problem_id: int) -> Optional[Tuple[str, str, str, str, str, str, str]]:
        """Fetch and process a single problem with explanations"""
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
                
                # Generate explanations in both languages and other content
                tasks = [
                    self.translate_to_mongolian(content),
                    self.generate_explanation(problem_id, title, content, self.answers[problem_id], "en"),
                    self.generate_explanation(problem_id, title, content, self.answers[problem_id], "mn")
                ]
                
                content_mn, explanation_en, explanation_mn = await asyncio.gather(*tasks)
                
                duration = time.time() - start_time
                self.timing_stats['fetch'].append(duration)
                logger.info(f"Completed problem {problem_id} in {duration:.2f} seconds")
                
                return (title, content, '', content_mn, explanation_en, explanation_mn)
                
        except Exception as e:
            logger.error(f"Error processing problem {problem_id}: {str(e)}")
        
        return None

    async def insert_problem(self, problem_id: int, problem_data: Tuple[str, str, str, str, str, str, str]) -> None:
        """Insert a problem into the database"""
        title, content, title_mn, content_mn, explanation_en, explanation_mn = problem_data
        try:
            self.pb.collection('problems').create({
                "description_en": content,
                "description_mn": content_mn,
                "answer": self.answers.get(problem_id, ""),
                "site": self.site_id,
                "title_en": title,
                "title_mn": title_mn,
                "order": problem_id,
                "explanation_en": explanation_en,
                "explanation_mn": explanation_mn
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
        await importer.import_problems(1, 1)
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())