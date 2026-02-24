#!/usr/bin/env python3
"""
Weather Search Agent - Playwright + LLM Client Integration

This script demonstrates integrating Playwright browser automation with our Bedrock LLM client
to search for today's weather and get an AI-powered summary.
"""

import asyncio
from datetime import datetime
from playwright.async_api import async_playwright
from llm.client import LLMClient, BedrockModel, MessageRole


class WeatherSearchAgent:
    """
    AI agent that uses browser automation to search for weather and provides intelligent summaries.
    """

    def __init__(self, model: BedrockModel = BedrockModel.CLAUDE_4_5_HAIKU):
        """Initialize the weather search agent."""
        self.llm_client = LLMClient(model=model)
        self.model = model
        self.browser = None
        self.page = None
        self.playwright = None

    async def __aenter__(self):
        """Async context manager entry - start browser."""
        await self.start_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup browser."""
        await self.close_browser()

    async def start_browser(self, headless: bool = False):
        """Start the Playwright browser."""
        print("🚀 Starting browser...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=headless,
            args=['--no-sandbox', '--disable-dev-shm-usage']  # Better compatibility
        )
        self.page = await self.browser.new_page()

        # Set a realistic user agent
        await self.page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        print("✅ Browser started successfully")

    async def close_browser(self):
        """Close the browser and cleanup."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        print("🔒 Browser closed")

    async def search_google_weather(self, location: str = "") -> dict:
        """
        Search for weather on Google.

        Args:
            location: Location to search for (empty for current location)

        Returns:
            Dictionary with search results and raw HTML
        """
        try:
            print(f"🔍 Searching for weather{f' in {location}' if location else ''}...")

            # Navigate to Google
            await self.page.goto("https://www.google.com", wait_until="networkidle")

            # Handle cookie consent if it appears
            try:
                await self.page.click('button:has-text("Accept all")', timeout=2000)
                print("📝 Accepted cookies")
            except:
                try:
                    await self.page.click('button:has-text("I agree")', timeout=2000)
                    print("📝 Accepted cookies")
                except:
                    print("ℹ️ No cookie consent found or already accepted")

            # Search for weather
            search_query = f"weather {location}".strip()
            search_box = await self.page.wait_for_selector('textarea[name="q"], input[name="q"]')
            await search_box.fill(search_query)
            await search_box.press("Enter")

            # Wait for results to load
            await self.page.wait_for_load_state("networkidle")

            # Try to find weather information in various possible locations
            weather_info = {}

            # Look for weather widget
            try:
                # Try different selectors for weather information
                weather_selectors = [
                    '[data-attrid="kc:/weather"]',  # Weather widget
                    '[data-ved*="weather"]',         # Weather results
                    '.wob_loc',                      # Location in weather
                    '.wob_tm',                       # Temperature
                    '.wob_dc',                       # Weather condition
                ]

                for selector in weather_selectors:
                    elements = await self.page.query_selector_all(selector)
                    if elements:
                        print(f"✅ Found weather elements with selector: {selector}")
                        break

                # Extract weather data
                temperature = await self.page.text_content('.wob_tm') or "N/A"
                condition = await self.page.text_content('.wob_dc') or "N/A"
                location_found = await self.page.text_content('.wob_loc') or "N/A"

                weather_info = {
                    "temperature": temperature,
                    "condition": condition,
                    "location": location_found,
                    "search_query": search_query
                }

                print(f"🌡️ Found weather: {temperature} - {condition} in {location_found}")

            except Exception as e:
                print(f"⚠️ Could not extract structured weather data: {e}")

            # Fallback: Get the page content for LLM processing
            page_content = await self.page.content()

            # Extract just the text content for easier LLM processing
            text_content = await self.page.evaluate("""
                () => {
                    // Remove scripts and styles
                    const scripts = document.querySelectorAll('script, style');
                    scripts.forEach(el => el.remove());

                    // Get text from weather-related elements first
                    const weatherElements = document.querySelectorAll(
                        '[data-attrid*="weather"], .wob_loc, .wob_tm, .wob_dc, .wob_dcp'
                    );

                    let weatherText = '';
                    weatherElements.forEach(el => {
                        weatherText += el.innerText + ' ';
                    });

                    // If we found weather text, prioritize it
                    if (weatherText.trim()) {
                        return 'Weather Information: ' + weatherText;
                    }

                    // Otherwise get general page text (first 3000 chars)
                    return document.body.innerText.substring(0, 3000);
                }
            """)

            return {
                "success": True,
                "weather_info": weather_info,
                "text_content": text_content,
                "search_query": search_query,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"❌ Error searching for weather: {e}")
            return {
                "success": False,
                "error": str(e),
                "search_query": search_query,
                "timestamp": datetime.now().isoformat()
            }

    async def analyze_weather_with_llm(self, search_results: dict) -> str:
        """
        Use the LLM client to analyze and summarize weather search results.

        Args:
            search_results: Results from the weather search

        Returns:
            AI-generated weather summary
        """
        try:
            print("🤖 Analyzing weather data with AI...")

            if not search_results["success"]:
                return f"I encountered an error while searching for weather: {search_results.get('error', 'Unknown error')}"

            # Create a prompt for the LLM
            system_prompt = """You are a helpful weather assistant. Analyze the provided weather search results
and give a clear, concise summary of the current weather conditions. Focus on:
1. Current temperature and conditions
2. Location
3. Any relevant details like humidity, wind, etc.
4. A brief outlook or recommendation

Keep it conversational and helpful. If the data seems incomplete, mention what information might be missing."""

            # Prepare the weather data for analysis
            weather_info = search_results.get("weather_info", {})
            text_content = search_results.get("text_content", "")

            user_prompt = f"""Please analyze this weather information and provide a helpful summary:

Search Query: {search_results.get('search_query', 'weather')}
Timestamp: {search_results.get('timestamp', 'unknown')}

Structured Data:
- Temperature: {weather_info.get('temperature', 'N/A')}
- Condition: {weather_info.get('condition', 'N/A')}
- Location: {weather_info.get('location', 'N/A')}

Raw Text from Search Results:
{text_content}

Please provide a natural, conversational weather summary based on this information."""

            # Create and submit prompt to LLM
            prompt = self.llm_client.create_prompt(
                prompt_text=user_prompt,
                system_prompt=system_prompt,
                model_override=self.model,
                max_token_output=300,
                temperature=0.3,
                prompt_name="weather-analysis"
            )

            response = self.llm_client.submit_prompt(prompt)

            if response.error:
                return f"AI analysis failed: {response.error}"

            print(f"💰 Analysis cost: ${response.cost:.6f}")
            print(f"📊 Tokens: {response.input_tokens}→{response.output_tokens}")

            return response.text

        except Exception as e:
            print(f"❌ Error during LLM analysis: {e}")
            return f"I encountered an error while analyzing the weather data: {str(e)}"

    async def get_weather_report(self, location: str = "", include_screenshot: bool = False) -> dict:
        """
        Complete weather search and analysis workflow.

        Args:
            location: Location to search for weather
            include_screenshot: Whether to take a screenshot

        Returns:
            Complete weather report with AI analysis
        """
        print("🌤️ Starting weather search and analysis...")
        print("=" * 60)

        # Search for weather
        search_results = await self.search_google_weather(location)

        # Take screenshot if requested
        screenshot_path = None
        if include_screenshot and search_results["success"]:
            try:
                screenshot_path = f"weather_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                await self.page.screenshot(path=screenshot_path)
                print(f"📸 Screenshot saved: {screenshot_path}")
            except Exception as e:
                print(f"⚠️ Could not save screenshot: {e}")

        # Analyze with LLM
        ai_summary = await self.analyze_weather_with_llm(search_results)

        return {
            "search_results": search_results,
            "ai_summary": ai_summary,
            "screenshot_path": screenshot_path,
            "timestamp": datetime.now().isoformat()
        }


async def main():
    """Main function to demonstrate the weather search agent."""
    print("🌤️ Weather Search Agent with Playwright + Bedrock LLM")
    print("=" * 80)

    try:
        # Create and run the weather agent
        async with WeatherSearchAgent(model=BedrockModel.CLAUDE_4_5_HAIKU) as agent:

            # Demo 1: Current location weather
            print("\n1️⃣ Searching for current location weather...")
            report1 = await agent.get_weather_report(include_screenshot=True)

            print("\n🤖 AI Weather Summary:")
            print("-" * 40)
            print(report1["ai_summary"])

            # Demo 2: Specific location weather
            print("\n\n2️⃣ Searching for weather in a specific location...")
            report2 = await agent.get_weather_report("San Francisco, CA", include_screenshot=False)

            print("\n🤖 AI Weather Summary:")
            print("-" * 40)
            print(report2["ai_summary"])

            # Summary
            print("\n" + "=" * 80)
            print("✅ Weather Search Agent Demo Complete!")
            print(f"🕒 Total time: {datetime.now().strftime('%H:%M:%S')}")

            if report1.get("screenshot_path"):
                print(f"📸 Screenshot saved: {report1['screenshot_path']}")

            print("\n🎯 Capabilities Demonstrated:")
            print("   ✅ Browser automation with Playwright")
            print("   ✅ Google search and data extraction")
            print("   ✅ AI analysis with Bedrock LLM")
            print("   ✅ Error handling and recovery")
            print("   ✅ Screenshot capture")
            print("   ✅ Cost tracking and performance metrics")

    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Starting Weather Search Agent...")
    asyncio.run(main())