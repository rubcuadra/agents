#!/usr/bin/env python3
"""
Simple Web Agent - Basic Playwright + LLM Integration Demo

A simplified demonstration of browser automation with AI analysis.
"""

import asyncio
from datetime import datetime
from playwright.async_api import async_playwright
from llm.client import LLMClient, BedrockModel


class SimpleWebAgent:
    """Simple web agent demonstrating Playwright + LLM integration."""

    def __init__(self, model: BedrockModel = BedrockModel.CLAUDE_4_5_HAIKU):
        self.llm_client = LLMClient(model=model)
        self.model = model

    async def search_and_analyze(self, search_term: str, site: str = "duckduckgo.com") -> dict:
        """
        Search for a term and get AI analysis of the results.

        Args:
            search_term: What to search for
            site: Search engine to use (default: DuckDuckGo for better bot-friendliness)
        """
        print(f"🔍 Searching for '{search_term}' on {site}")
        print("=" * 60)

        async with async_playwright() as playwright:
            # Launch browser
            browser = await playwright.chromium.launch(headless=True)  # Headless for demo
            page = await browser.new_page()

            try:
                # Navigate to search engine
                if site == "duckduckgo.com":
                    await page.goto("https://duckduckgo.com")
                    await page.fill('[name="q"]', search_term)
                    await page.press('[name="q"]', "Enter")
                else:
                    # Fallback to direct URL
                    await page.goto(f"https://{site}")

                # Wait for page to load
                await page.wait_for_load_state("networkidle")

                # Extract page content
                title = await page.title()

                # Get text content (first 2000 characters)
                content = await page.evaluate("""
                    () => {
                        // Remove scripts and styles
                        const scripts = document.querySelectorAll('script, style');
                        scripts.forEach(el => el.remove());

                        return document.body.innerText.substring(0, 2000);
                    }
                """)

                # Take screenshot
                screenshot_path = f"web_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                await page.screenshot(path=screenshot_path)

                print(f"✅ Page loaded: {title}")
                print(f"📸 Screenshot: {screenshot_path}")

                search_results = {
                    "success": True,
                    "title": title,
                    "content": content,
                    "screenshot": screenshot_path,
                    "search_term": search_term,
                    "site": site
                }

            except Exception as e:
                print(f"❌ Error during search: {e}")
                search_results = {
                    "success": False,
                    "error": str(e),
                    "search_term": search_term,
                    "site": site
                }

            finally:
                await browser.close()

        # Analyze with AI
        if search_results["success"]:
            ai_analysis = await self.analyze_content_with_ai(search_results)
            search_results["ai_analysis"] = ai_analysis

        return search_results

    async def analyze_content_with_ai(self, search_data: dict) -> str:
        """Use LLM to analyze the web content."""
        print("\n🤖 Analyzing content with AI...")

        system_prompt = """You are a helpful assistant that analyzes web search results.
Provide a clear, concise summary of the key information found. Focus on:
1. Main topics or themes
2. Key facts or data points
3. Relevant insights or trends
4. A brief, helpful conclusion

Keep your analysis conversational and informative."""

        user_prompt = f"""Please analyze this web search content:

Search Term: {search_data['search_term']}
Page Title: {search_data['title']}
Website: {search_data['site']}

Content:
{search_data['content']}

Please provide a helpful summary and analysis of what was found."""

        try:
            prompt = self.llm_client.create_prompt(
                prompt_text=user_prompt,
                system_prompt=system_prompt,
                model_override=self.model,
                max_token_output=400,
                temperature=0.4,
                prompt_name="web-content-analysis"
            )

            response = self.llm_client.submit_prompt(prompt)

            if response.error:
                return f"AI analysis failed: {response.error}"

            print(f"💰 Analysis cost: ${response.cost:.6f}")
            print(f"📊 Tokens: {response.input_tokens}→{response.output_tokens}")

            return response.text

        except Exception as e:
            return f"Error during AI analysis: {str(e)}"


async def demo_web_agent():
    """Demonstrate the web agent capabilities."""
    print("🌐 Simple Web Agent Demo - Playwright + Bedrock LLM")
    print("=" * 80)

    agent = SimpleWebAgent(model=BedrockModel.CLAUDE_4_5_HAIKU)

    # Demo searches
    demos = [
        ("today's weather", "duckduckgo.com"),
        ("latest technology news", "duckduckgo.com"),
        ("Python programming tips", "duckduckgo.com"),
    ]

    for search_term, site in demos:
        print(f"\n🔍 Demo: Searching for '{search_term}'")

        try:
            results = await agent.search_and_analyze(search_term, site)

            if results["success"]:
                print("\n🤖 AI Analysis:")
                print("-" * 50)
                print(results["ai_analysis"])
                print(f"\n📸 Screenshot saved: {results['screenshot']}")
            else:
                print(f"❌ Search failed: {results.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Demo failed: {e}")

        print("\n" + "=" * 80)

    print("✅ Demo complete! Check the generated screenshots.")


if __name__ == "__main__":
    print("🚀 Starting Simple Web Agent Demo...")
    asyncio.run(demo_web_agent())