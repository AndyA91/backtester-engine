import asyncio
import re
import os
import sys
from pathlib import Path
from playwright.async_api import async_playwright

# Configuration
URL = "https://www.tradingview.com/u/HPotter/#published-scripts"
MIN_BOOSTS = 150          # Minimum likes/boosts required
INDICATORS_ONLY = True    # Skip strategies

# Extract author from URL for organized storage
AUTHOR_NAME = URL.split("/u/")[1].split("/")[0]
BASE_OUTPUT_DIR = Path("indicators")
OUTPUT_DIR = BASE_OUTPUT_DIR / AUTHOR_NAME

# Stylistic Groups (Enable/Disable by adding to ACTIVE_GROUPS)
STYLE_GROUPS = {
    "Trend & Momentum": ["Trend", "Momentum", "Squeeze", "EMA", "SMA", "Crossover", "Oscillator", "Pivot"],
    "Volatility & Breakouts": ["Bollinger", "ATR", "Keltner", "Donchian", "Breakout", "Band", "Channel"],
    "Mean Reversion": ["RSI", "Stochastic", "CCI", "Mean", "Reversion", "Overbought", "Oversold", "Divergence"],
    "SMC & Liquidity": ["Order Block", "FVG", "Fair Value", "Liquidity", "Structure", "SMC", "Supply", "Demand", "Imbalance"],
    "Volume & Flow": ["OBV", "CMF", "Volume", "Money Flow", "Accumulation", "Distribution", "Flow"],
    "Adaptive & Filters": ["KAMA", "Jurik", "Ehlers", "Gaussian", "SuperSmooth", "Butterworth", "Adaptive"]
}

# Set to [] to grab everything, or list specific groups from above
ACTIVE_GROUPS = list(STYLE_GROUPS.keys())

# Static Analysis Forbidden Patterns (Regex)
FORBIDDEN_PATTERNS = [
    r"request\.security",
    r"security\(",
    r"lookahead\s*=\s*barmerge\.lookahead_on",
    r"calc_on_every_tick\s*=\s*true",
    r"calc_on_order_fills\s*=\s*true",
    r"varip\s+"
]

async def scrape_tv_author():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False) # Headless=False for debugging/visiblity
        page = await browser.new_page()
        
        print(f"Navigating to {URL}...")
        await page.goto(URL, wait_until="networkidle")
        await asyncio.sleep(2) # Extra buffer for dynamic content

        # Scroll to load all scripts
        print("Scrolling to load all scripts...")
        while True:
            last_count = len(await page.query_selector_all('article[class*="card-"]'))
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(4) # Increased for stability
            new_count = len(await page.query_selector_all('article[class*="card-"]'))
            print(f"  Loaded {new_count} scripts...")
            if new_count == last_count:
                # Try one more scroll just in case
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(4)
                if len(await page.query_selector_all('article[class*="card-"]')) == new_count:
                    break

        # Extract Script Cards
        cards = await page.query_selector_all('article[class*="card-"]')
        print(f"Found {len(cards)} total scripts. Filtering...")

        scripts_to_fetch = []
        for card in cards:
            title_el = await card.query_selector('a[data-qa-id="ui-lib-card-link-title"]')
            if not title_el: continue
            
            title = await title_el.inner_text()
            href = await title_el.get_attribute("href")
            if href.startswith("http"):
                link = href
            else:
                link = f"https://www.tradingview.com{href}"

            # Boost Count
            # The count is in the aria-label of a child span or digit spans
            boost_count = 0
            boost_label_el = await card.query_selector('[aria-label*="boosts"]')
            if boost_label_el:
                aria_label = await boost_label_el.get_attribute("aria-label") or ""
                match = re.search(r"(\d+)", aria_label)
                if match:
                    boost_count = int(match.group(1))
            else:
                # Fallback: check the button itself
                boost_btn = await card.query_selector('[data-qa-id="ui-lib-card-like-button"]')
                if boost_btn:
                    aria_label = await boost_btn.get_attribute("aria-label") or ""
                    match = re.search(r"(\d+)", aria_label)
                    if match:
                        boost_count = int(match.group(1))

            # Category (Indicator vs Strategy)
            category_el = await card.query_selector('span[class*="visuallyHiddenLabel-"]')
            category = await category_el.inner_text() if category_el else ""
            is_strategy = "strategy" in category.lower()

            # --- Apply Metadata Filters ---
            if boost_count < MIN_BOOSTS:
                print(f"  [Skip] {title} ({boost_count} boosts < {MIN_BOOSTS})")
                continue
            
            if INDICATORS_ONLY and is_strategy:
                print(f"  [Skip] {title} (is strategy)")
                continue

            if ACTIVE_GROUPS:
                matched_kw = None
                for group in ACTIVE_GROUPS:
                    keywords = STYLE_GROUPS.get(group, [])
                    for kw in keywords:
                        if kw.lower() in title.lower():
                            matched_kw = kw
                            break
                    if matched_kw: break
                
                if not matched_kw:
                    print(f"  [Skip] {title} (no keyword match in active groups)")
                    continue
                else:
                    print(f"  [Match] {title} (Keyword: '{matched_kw}')")

            scripts_to_fetch.append({"title": title, "link": link, "boosts": boost_count})

        print(f"Filtered down to {len(scripts_to_fetch)} interesting scripts. Downloading source...")

        # Process Each Filtered Script
        for script in scripts_to_fetch:
            print(f"Fetching: {script['title']} ({script['boosts']} boosts)...")
            try:
                await page.goto(script['link'], wait_until="networkidle")
                await asyncio.sleep(3)
                
                # 1. Click the 'Source code' tab (id="code")
                # Sometimes it's a link, sometimes a button
                code_tab = await page.query_selector('#code')
                if code_tab:
                    await code_tab.click()
                    await asyncio.sleep(2) # Wait for code to render
                
                # 2. Extract the source code
                # The subagent found 'div[class*="code-"]' to be the cleanest container
                source_box = await page.query_selector('div[class*="code-G0eoc001"]') or \
                             await page.query_selector('div[class*="code-"]') or \
                             await page.query_selector('.tv-community-scripts__source-code-inner')
                
                if not source_box:
                    print(f"  [Error] Could not find source code for {script['title']}")
                    # Debug: take a screenshot if it fails
                    # await page.screenshot(path=f"debug_{safe_name}.png")
                    continue

                source_code = await source_box.inner_text()

                # --- Apply Static Analysis Filter ---
                too_complex = False
                for pattern in FORBIDDEN_PATTERNS:
                    if re.search(pattern, source_code, re.IGNORECASE):
                        print(f"  [Skipped] Contains forbidden pattern: {pattern}")
                        too_complex = True
                        break
                
                if too_complex:
                    continue

                # Save to file
                safe_name = re.sub(r'[^\w\-_\. ]', '_', script['title']).replace(' ', '_')
                filename = OUTPUT_DIR / f"{safe_name}.pine"
                
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"// Title: {script['title']}\n")
                    f.write(f"// Boosts: {script['boosts']}\n")
                    f.write(f"// Link: {script['link']}\n")
                    f.write("// Scraped via tools/tv_scraper.py\n\n")
                    f.write(source_code)
                
                print(f"  [Saved] -> {filename}")

            except Exception as e:
                print(f"  [Error] Failed to process {script['title']}: {e}")

        await browser.close()
        print("\nScraping Complete.")

if __name__ == "__main__":
    asyncio.run(scrape_tv_author())
