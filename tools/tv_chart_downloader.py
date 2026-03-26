"""
TradingView Chart Data Downloader

Automates chart data export from TradingView using Playwright.
Default: Renko charts. Also supports candle, Heikin Ashi, etc.

Usage:
    python tools/tv_chart_downloader.py

Workflow:
    1. Opens browser (visible) — you log in manually
    2. For each job: sets symbol, timeframe, chart type, brick size
    3. Scrolls left to load all bars, then exports CSV to data/

Selectors last verified: 2026-03-25
"""

import asyncio
import sys
import threading
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# Fix Windows console Unicode encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── Configuration ────────────────────────────────────────────────────────────

CHART_URL = "https://www.tradingview.com/chart/"
OUTPUT_DIR = Path("data")
SCREENSHOT_DIR = Path("tools/debug_screenshots")

# Each job = one chart data export.
JOBS = [
    {"symbol": "OANDA:USDCHF", "chart_type": "renko", "timeframe": "1s", "brick_size": "0.001"},
    {"symbol": "OANDA:USDCHF", "chart_type": "renko", "timeframe": "1s", "brick_size": "0.0011"},
    {"symbol": "OANDA:USDCHF", "chart_type": "renko", "timeframe": "1s", "brick_size": "0.0012"},
    {"symbol": "OANDA:USDCHF", "chart_type": "renko", "timeframe": "1s", "brick_size": "0.0013"},
    {"symbol": "OANDA:USDCHF", "chart_type": "renko", "timeframe": "1s", "brick_size": "0.0014"},
    {"symbol": "OANDA:USDCHF", "chart_type": "renko", "timeframe": "1s", "brick_size": "0.0015"},
]

# Map timeframe shorthand → aria-label on toolbar buttons
TIMEFRAME_LABELS = {
    "1s": "1 second", "5s": "5 seconds", "10s": "10 seconds",
    "15s": "15 seconds", "30s": "30 seconds",
    "1m": "1 minute", "2m": "2 minutes", "3m": "3 minutes",
    "5m": "5 minutes", "10m": "10 minutes", "15m": "15 minutes",
    "30m": "30 minutes", "D": "1 day",
}


# ─── GUI prompt ───────────────────────────────────────────────────────────────

def gui_prompt(message: str, title: str = "TradingView Downloader"):
    """Show a blocking GUI popup. Works from any shell (no stdin needed)."""
    def _show():
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showinfo(title, message, parent=root)
        root.destroy()
    t = threading.Thread(target=_show, daemon=True)
    t.start()
    t.join()


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def debug_screenshot(page, name: str):
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_DIR / f"{name}.png"
    await page.screenshot(path=str(path.resolve()))
    print(f"    [Debug] Screenshot: {path}")


async def dismiss_overlays(page):
    """Dismiss popups, cookie banners, error overlays."""
    for text in ["Accept all", "Accept", "Got it", "Maybe later", "Not now"]:
        try:
            btn = page.get_by_role("button", name=text).first
            if await btn.count() > 0 and await btn.is_visible():
                await btn.click()
                await asyncio.sleep(0.3)
        except Exception:
            pass
    # Error card overlays
    try:
        cards = page.locator('[class*="errorCard"]')
        for i in range(await cards.count()):
            close = cards.nth(i).locator("button").first
            if await close.count() > 0:
                await close.click()
                await asyncio.sleep(0.3)
    except Exception:
        pass
    # Generic close
    for sel in ['button[aria-label="Close"]', '[data-name="close"]']:
        try:
            btn = page.locator(sel).first
            if await btn.count() > 0 and await btn.is_visible():
                await btn.click()
                await asyncio.sleep(0.3)
        except Exception:
            pass


# ─── Chart configuration steps ───────────────────────────────────────────────

async def wait_for_login(page):
    print("Opening TradingView chart...")
    await page.goto(CHART_URL, wait_until="domcontentloaded")
    await asyncio.sleep(4)
    await dismiss_overlays(page)
    gui_prompt(
        "Log in to TradingView in the browser window.\n\n"
        "Once logged in and the chart is visible, click OK."
    )
    await dismiss_overlays(page)
    await asyncio.sleep(1)


async def set_symbol(page, symbol: str):
    """Click the symbol button (#header-toolbar-symbol-search), type symbol, press Enter."""
    print(f"  Setting symbol: {symbol}")

    # Click the symbol search button (confirmed selector)
    btn = page.locator("#header-toolbar-symbol-search").first
    try:
        await btn.wait_for(state="visible", timeout=5000)
        await btn.click()
    except PlaywrightTimeout:
        gui_prompt(f"Could not find symbol button.\n\nSet symbol to {symbol} manually, then click OK.")
        return

    await asyncio.sleep(1)

    # The search dialog opens — find the text input
    # Try common search input patterns
    search = None
    for sel in ['input[data-role="search"]', 'input[placeholder*="Search"]',
                'input[placeholder*="search"]', 'input[placeholder*="Symbol"]',
                'input[type="text"]']:
        loc = page.locator(sel).first
        try:
            if await loc.count() > 0 and await loc.is_visible():
                search = loc
                break
        except Exception:
            continue

    if not search:
        # Fallback: just start typing — the dialog might auto-focus an input
        print("    Typing symbol directly (no input found)...")
        await page.keyboard.type(symbol, delay=30)
        await asyncio.sleep(2)
        await page.keyboard.press("Enter")
        await asyncio.sleep(3)
        return

    await search.fill("")
    await search.type(symbol, delay=30)
    await asyncio.sleep(2)
    await search.press("Enter")
    await asyncio.sleep(3)
    await dismiss_overlays(page)


async def set_timeframe(page, timeframe: str):
    """Click the direct timeframe button in the toolbar (e.g. '1s' → aria-label='1 second')."""
    label = TIMEFRAME_LABELS.get(timeframe.lower())
    print(f"  Setting timeframe: {timeframe} (label: {label})")

    if label:
        # Direct toolbar button — fastest & most reliable
        # Use .last because TV has 3 duplicate toolbars; the real one (with IDs) is last
        btn = page.locator(f'button[aria-label="{label}"]').last
        try:
            if await btn.count() > 0 and await btn.is_visible():
                await btn.click()
                await asyncio.sleep(2)
                return
        except Exception:
            pass

    # Fallback: open the interval dropdown
    dropdown = page.locator('button[aria-label="Chart interval"]').first
    try:
        if await dropdown.count() > 0 and await dropdown.is_visible():
            await dropdown.click()
            await asyncio.sleep(1)
            # Type in search or click matching option
            opt = page.get_by_text(timeframe, exact=True).first
            if await opt.count() > 0:
                await opt.click()
                await asyncio.sleep(2)
                return
    except Exception:
        pass

    gui_prompt(f"Set timeframe to {timeframe}, then click OK.")


async def set_chart_type(page, chart_type: str):
    """Open chart type dropdown (#header-toolbar-chart-styles) and click the type."""
    type_labels = {
        "candle": "Candles", "renko": "Renko", "heikin_ashi": "Heikin Ashi",
        "hollow_candle": "Hollow candles", "line": "Line", "area": "Area",
        "baseline": "Baseline", "range": "Range",
    }
    label = type_labels.get(chart_type, chart_type)
    print(f"  Setting chart type: {label}")

    # Open chart type dropdown (confirmed: #header-toolbar-chart-styles works,
    # but the aria-label changes to match current type — so use id)
    btn = page.locator("#header-toolbar-chart-styles").first
    opened = False
    try:
        if await btn.count() > 0 and await btn.is_visible():
            await btn.click()
            opened = True
    except Exception:
        pass

    if not opened:
        # Fallback: the button's aria-label is the current chart type name
        # e.g. aria-label="Candles" or aria-label="Renko"
        for current_type in type_labels.values():
            try:
                alt = page.locator(f'button[aria-label="{current_type}"]').first
                if await alt.count() > 0 and await alt.is_visible():
                    await alt.click()
                    opened = True
                    break
            except Exception:
                continue

    if not opened:
        gui_prompt(f"Open the chart type dropdown and select {label}.\n\nClick OK when done.")
        return

    await asyncio.sleep(1)

    # Click the target type in the dropdown
    # The dropdown items have text content matching the type label
    clicked = False
    try:
        # Use JS to find the menu item — more reliable than text matching
        clicked = await page.evaluate(f"""() => {{
            const items = document.querySelectorAll('[class*="menu"] *, [role="option"], [role="menuitem"], [role="listbox"] *');
            for (const el of items) {{
                if (el.textContent?.trim() === '{label}' && el.offsetParent !== null) {{
                    el.click();
                    return true;
                }}
            }}
            return false;
        }}""")
    except Exception:
        pass

    if not clicked:
        try:
            opt = page.get_by_text(label, exact=True).first
            if await opt.count() > 0 and await opt.is_visible():
                await opt.click()
                clicked = True
        except Exception:
            pass

    if not clicked:
        gui_prompt(f"Select '{label}' from the dropdown.\n\nClick OK when done.")

    await asyncio.sleep(2)


async def set_renko_brick_size(page, brick_size: str):
    """Open chart settings (#header-toolbar-properties) and set the Renko box size."""
    print(f"  Setting brick size: {brick_size}")

    # Open settings dialog (confirmed: id="header-toolbar-properties", aria-label="Settings")
    settings_btn = page.locator("#header-toolbar-properties").first
    opened = False
    try:
        if await settings_btn.count() > 0 and await settings_btn.is_visible():
            await settings_btn.click()
            opened = True
    except Exception:
        pass

    if not opened:
        gui_prompt(f"Open Chart Settings and set box size to {brick_size}.\n\nClick OK when done.")
        return

    # Wait for settings dialog to fully render — retry up to 8s
    dialog_ready = False
    for wait_attempt in range(4):
        try:
            await page.get_by_text("Symbol", exact=True).first.wait_for(state="visible", timeout=2000)
            dialog_ready = True
            break
        except PlaywrightTimeout:
            await asyncio.sleep(1)
    if dialog_ready:
        print("    Settings dialog opened")
    else:
        print("    [Warn] Settings dialog slow — proceeding anyway")
        await asyncio.sleep(2)

    # Navigate to the "Symbol" tab
    try:
        symbol_tab = page.get_by_text("Symbol", exact=True).first
        if await symbol_tab.count() > 0 and await symbol_tab.is_visible():
            await symbol_tab.click()
            await asyncio.sleep(1)
    except Exception:
        pass

    # Step 1: Change "Box size assignment method" from ATR -> Traditional
    # Retry loop: the dropdown element can take time to appear after tab switch
    method_changed = False
    already_traditional = False

    for retry in range(3):
        # First check if already set to "Traditional"
        try:
            trad_check = page.get_by_text("Traditional", exact=True)
            if await trad_check.count() > 0:
                for idx in range(await trad_check.count()):
                    el = trad_check.nth(idx)
                    if await el.is_visible():
                        parent_text = await el.locator("xpath=..").text_content()
                        if parent_text and "length" not in parent_text.lower():
                            already_traditional = True
                            print("    Box size method already Traditional")
                            break
        except Exception:
            pass

        if already_traditional:
            break

        # Try to find and click the "ATR" dropdown value
        try:
            atr_elements = page.get_by_text("ATR", exact=True)
            count = await atr_elements.count()
            for idx in range(count):
                el = atr_elements.nth(idx)
                if await el.is_visible():
                    parent_text = await el.locator("xpath=..").text_content()
                    if parent_text and "length" not in parent_text.lower():
                        await el.click()
                        method_changed = True
                        print("    Opened box size method dropdown")
                        break
        except Exception as e:
            print(f"    [Debug] ATR dropdown attempt {retry+1}: {e}")

        if method_changed:
            break
        # Wait and retry
        await asyncio.sleep(1.5)

    if method_changed:
        await asyncio.sleep(1)
        trad_clicked = False
        try:
            trad = page.get_by_text("Traditional", exact=True).first
            if await trad.count() > 0 and await trad.is_visible():
                await trad.click()
                trad_clicked = True
                print("    Changed box size method: ATR -> Traditional")
        except Exception:
            pass
        if not trad_clicked:
            print("    [Warn] Could not select 'Traditional' -- trying manual")
            gui_prompt("Select 'Traditional' from the 'Box size assignment method' dropdown.\n\nClick OK when done.")
        await asyncio.sleep(1)
    elif not already_traditional:
        print("    [Warn] ATR dropdown not found after retries")
        gui_prompt("Change 'Box size assignment method' to 'Traditional'.\n\nClick OK when done.")

    # Step 2: Now find and fill the "Box size" input (appears after switching to Traditional)
    await asyncio.sleep(0.5)
    set_ok = False
    try:
        # Find "Box size" label (exact, not "Box size assignment method")
        box_labels = page.get_by_text("Box size", exact=True)
        count = await box_labels.count()
        for idx in range(count):
            label = box_labels.nth(idx)
            if await label.is_visible():
                # Find the input in the same row — walk up to find a container with an input
                # Try parent, grandparent, etc.
                for depth in range(1, 6):
                    ancestor = label.locator(f"xpath={'/..' * depth}")
                    inp = ancestor.locator('input:not([type="checkbox"]):not([type="radio"]):not([type="hidden"])').first
                    if await inp.count() > 0 and await inp.is_visible():
                        await inp.click(click_count=3)  # Select all
                        await inp.fill(brick_size)
                        # Also press Tab to confirm the value
                        await inp.press("Tab")
                        set_ok = True
                        print(f"    Box size set to {brick_size}")
                        break
                if set_ok:
                    break
    except Exception as e:
        print(f"    [Debug] Box size input fill failed: {e}")

    if not set_ok:
        print("    [Warn] Could not find box size input")
        await debug_screenshot(page, "brick_size_fail")
        gui_prompt(f"Set the box size to {brick_size} in the settings dialog.\n\nClick OK when done.")

    # Click OK to close settings
    await asyncio.sleep(0.5)
    closed = False
    for btn_text in ["OK", "Apply"]:
        try:
            btn = page.get_by_role("button", name=btn_text).first
            if await btn.count() > 0 and await btn.is_visible():
                await btn.click()
                closed = True
                break
        except Exception:
            continue
    if not closed:
        await page.keyboard.press("Enter")

    # Wait for chart to reload with new brick size
    print("    Waiting for chart to reload...")
    await asyncio.sleep(5)


async def enable_autofit(page):
    """Enable auto-scale/autofit so the chart fits data to screen."""
    try:
        auto_btn = page.locator('button[aria-label="Toggle auto scale"]').first
        if await auto_btn.count() > 0 and await auto_btn.is_visible():
            await auto_btn.click()
            await asyncio.sleep(0.5)
            print("    Autofit toggled")
    except Exception:
        pass


async def zoom_out_chart(page, drags: int = 10):
    """Zoom out by dragging the time axis (bottom bar) to the left."""
    print(f"    Zooming out (dragging time axis, {drags}x)...")

    # The time axis is at the very bottom of the chart canvas area.
    # Find the chart container and target the last ~30px (time axis strip).
    canvas = page.locator('canvas[data-name="pane-canvas"]').first
    try:
        box = await canvas.bounding_box()
        if not box:
            print("    [Warn] Could not find chart canvas for zoom")
            return
    except Exception:
        return

    # Time axis is below the chart canvas — typically ~20-30px below canvas bottom
    time_axis_y = box["y"] + box["height"] + 15
    mid_x = box["x"] + box["width"] / 2
    drag_amount = int(box["width"] * 0.3)

    for i in range(drags):
        # Drag from center-left to center-right on time axis = zoom out
        start_x = mid_x - drag_amount / 2
        end_x = mid_x + drag_amount / 2
        await page.mouse.move(start_x, time_axis_y)
        await page.mouse.down()
        await page.mouse.move(end_x, time_axis_y, steps=5)
        await page.mouse.up()
        await asyncio.sleep(0.3)

    await asyncio.sleep(1)
    print("    Zoom out complete")


async def focus_chart(page):
    """Click the center of the chart to ensure keyboard focus is on it."""
    # Try clicking the chart canvas directly at its center
    canvas = page.locator('canvas[data-name="pane-canvas"]').first
    try:
        box = await canvas.bounding_box()
        if box:
            # Click in the middle of the chart using page.mouse (not locator.click)
            # This is more reliable for focus than locator.click(force=True)
            await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            await asyncio.sleep(0.5)
            return True
    except Exception:
        pass
    return False


async def get_chart_date_range(page):
    """Try to read the date range text from the chart to detect stall."""
    try:
        # The date range appears in the bottom-left of the chart (time axis)
        # Use the go-to-date button's nearby text, or read bottom toolbar
        info = await page.evaluate("""() => {
            // Look for the leftmost visible date on the time axis
            const labels = document.querySelectorAll('[class*="price-axis"] *, [class*="time-axis"] *');
            let texts = [];
            for (const el of labels) {
                const t = el.textContent?.trim();
                if (t && t.length > 4 && t.length < 30) texts.push(t);
            }
            return texts.slice(0, 5).join(' | ');
        }""")
        return info
    except Exception:
        return ""


async def scroll_to_load_all_bars(page):
    """Zoom out + drag-scroll chart left to load all historical bars."""
    print("  Loading all bars...")

    await dismiss_overlays(page)
    await enable_autofit(page)
    await asyncio.sleep(0.5)

    # Zoom out first to cover more history per drag
    await zoom_out_chart(page, drags=5)

    canvas = page.locator('canvas[data-name="pane-canvas"]').first
    box = await canvas.bounding_box()
    if not box:
        print("    [Warn] Could not find chart canvas")
        return

    cy = box["y"] + box["height"] / 2
    drag_distance = int(box["width"] * 0.7)

    # Drag-scroll left with stall detection
    print("    Drag-scrolling left...")
    prev_screenshot = None
    stall_count = 0
    max_drags = 28

    for attempt in range(max_drags):
        start_x = box["x"] + box["width"] * 0.15
        end_x = start_x + drag_distance

        await page.mouse.move(start_x, cy)
        await page.mouse.down()
        await page.mouse.move(end_x, cy, steps=5)
        await page.mouse.up()
        await asyncio.sleep(0.8)

        if attempt > 0 and attempt % 10 == 0:
            print(f"    ... drag {attempt}/{max_drags}")

        # Stall detection: take a quick screenshot hash every 5 drags
        if attempt > 0 and attempt % 5 == 0:
            try:
                shot = await page.screenshot()
                if prev_screenshot and shot == prev_screenshot:
                    stall_count += 1
                    if stall_count >= 2:
                        print(f"    Chart stalled after {attempt} drags (fully loaded)")
                        break
                else:
                    stall_count = 0
                prev_screenshot = shot
            except Exception:
                pass

    # Return to latest bar
    await focus_chart(page)
    await asyncio.sleep(0.3)
    await page.keyboard.press("End")
    await asyncio.sleep(2)
    print("  All bars loaded.")


async def find_export_button(page):
    """Scan toolbar for the export/download button and print what we find."""
    # Dump all toolbar buttons to help identify the export button
    buttons_info = await page.evaluate("""() => {
        const btns = document.querySelectorAll('button');
        return Array.from(btns)
            .filter(b => b.offsetParent !== null && b.getBoundingClientRect().y < 140)
            .map(b => ({
                text: b.textContent?.trim().substring(0, 40),
                ariaLabel: b.getAttribute('aria-label'),
                dataName: b.getAttribute('data-name'),
                title: b.getAttribute('title'),
                x: Math.round(b.getBoundingClientRect().x),
                y: Math.round(b.getBoundingClientRect().y),
                w: Math.round(b.getBoundingClientRect().width),
            }));
    }""")
    print("    [Debug] Toolbar buttons:")
    for b in buttons_info:
        parts = []
        if b.get("dataName"): parts.append(f"data-name='{b['dataName']}'")
        if b.get("ariaLabel"): parts.append(f"aria='{b['ariaLabel']}'")
        if b.get("title"): parts.append(f"title='{b['title']}'")
        if b.get("text"): parts.append(f"text='{b['text'][:30]}'")
        if parts:
            print(f"      x={b['x']:4d} y={b['y']:3d} w={b['w']:3d} | {' | '.join(parts)}")
    return buttons_info


async def export_chart_data(page, output_path: Path):
    """Export chart data — tries multiple menu paths to find 'Export chart data'."""
    print(f"  Exporting -> {output_path.name}")
    await dismiss_overlays(page)

    export_found = False

    # Strategy 1: Right-click on the chart canvas (fastest, most reliable)
    try:
        canvas = page.locator('canvas[data-name="pane-canvas"]').first
        if await canvas.count() > 0:
            box = await canvas.bounding_box()
            if box:
                await page.mouse.click(
                    box["x"] + box["width"] / 2,
                    box["y"] + box["height"] / 2,
                    button="right",
                )
                await asyncio.sleep(1.5)
                opt = page.get_by_text("chart data", exact=False).first
                if await opt.count() > 0 and await opt.is_visible():
                    await opt.click()
                    export_found = True
                    print("    Found via right-click context menu")
                else:
                    await page.keyboard.press("Escape")
                    await asyncio.sleep(0.3)
    except Exception as e:
        print(f"    [Debug] Right-click approach: {e}")

    # Strategy 2: Toolbar buttons — try known selectors with force=True
    if not export_found:
        export_selectors = [
            'button[data-name="save-load-menu"]',   # dropdown arrow next to Save
            'button[aria-label="Manage layouts"]',   # same button, alt selector
            'button[aria-label*="xport"]',
            'button[aria-label*="ownload"]',
            'button[data-name*="export"]',
            'button[data-name*="download"]',
        ]
        for sel in export_selectors:
            if export_found:
                break
            try:
                btns = page.locator(sel)
                count = await btns.count()
                for idx in range(count):
                    btn = btns.nth(idx)
                    if await btn.is_visible():
                        await btn.click(force=True)
                        await asyncio.sleep(1)
                        opt = page.get_by_text("chart data", exact=False).first
                        if await opt.count() > 0 and await opt.is_visible():
                            await opt.click()
                            export_found = True
                            print(f"    Found via toolbar: {sel}")
                            break
                        await page.keyboard.press("Escape")
                        await asyncio.sleep(0.3)
            except Exception:
                pass

    # Strategy 3: Click "More" button near chart legend (force=True to bypass overlay)
    if not export_found:
        try:
            more_btns = page.locator('button[aria-label="More"]')
            count = await more_btns.count()
            for idx in range(count):
                btn = more_btns.nth(idx)
                if await btn.is_visible():
                    bx = await btn.bounding_box()
                    if bx and 30 < bx["y"] < 200:
                        await btn.click(force=True)
                        await asyncio.sleep(1)
                        opt = page.get_by_text("chart data", exact=False).first
                        if await opt.count() > 0 and await opt.is_visible():
                            await opt.click()
                            export_found = True
                            break
                        await page.keyboard.press("Escape")
                        await asyncio.sleep(0.3)
        except Exception as e:
            print(f"    [Debug] More button: {e}")

    # Strategy 4: Dump toolbar buttons for debugging, then ask user
    if not export_found:
        await find_export_button(page)
        gui_prompt("Click the down-arrow next to Save in toolbar,\n"
                   "then click 'Download chart data'.\n\nClick OK when the download dialog appears.")

    await asyncio.sleep(1.5)

    # Wait for Download button to be ready (TV shows a spinner while preparing)
    dl_btn = page.get_by_role("button", name="Download").first
    for wait in range(20):  # up to ~30s
        try:
            if await dl_btn.count() > 0 and await dl_btn.is_visible():
                btn_text = await dl_btn.text_content()
                if btn_text and "Download" in btn_text:
                    break
        except Exception:
            pass
        await asyncio.sleep(1.5)
    else:
        print("    [Warn] Download button not ready after 30s")

    # Click Download button and capture the download
    try:
        async with page.expect_download(timeout=30000) as download_info:
            if await dl_btn.count() > 0 and await dl_btn.is_visible():
                await dl_btn.click()
            else:
                # Fallback: try "Export" in case TV changes the label
                exp_btn = page.get_by_role("button", name="Export").first
                if await exp_btn.count() > 0 and await exp_btn.is_visible():
                    await exp_btn.click()

        download = await download_info.value
        await download.save_as(str(output_path.resolve()))
        print(f"  [OK] Saved: {output_path}")

    except PlaywrightTimeout:
        print("    [Error] Download timed out")
        await debug_screenshot(page, "export_timeout")
    except Exception as e:
        print(f"    [Error] Export failed: {e}")
        await debug_screenshot(page, "export_error")


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("TradingView Chart Data Downloader")
    print(f"Jobs: {len(JOBS)} | Output: {OUTPUT_DIR.resolve()}")
    print("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        await page.set_viewport_size({"width": 1280, "height": 720})

        async def _safe_dismiss(dialog):
            try:
                await dialog.accept()
            except Exception:
                pass
        page.on("dialog", lambda d: asyncio.ensure_future(_safe_dismiss(d)))

        await wait_for_login(page)

        succeeded = 0
        prev_symbol = None
        prev_chart_type = None
        prev_timeframe = None

        for i, job in enumerate(JOBS, 1):
            symbol = job["symbol"]
            chart_type = job.get("chart_type", "candle")
            timeframe = job.get("timeframe", "1s")
            brick_size = job.get("brick_size")

            header = f"[{i}/{len(JOBS)}] {symbol} | {chart_type} | tf={timeframe}"
            if brick_size:
                header += f" | brick={brick_size}"
            print(f"\n{header}")
            print("-" * len(header))

            # Check if browser is still alive — abort if closed
            try:
                await page.evaluate("1")
            except Exception:
                print("  [Fatal] Browser closed — aborting remaining jobs.")
                break

            try:
                if symbol != prev_symbol:
                    await set_symbol(page, symbol)
                    prev_symbol = symbol

                if timeframe != prev_timeframe:
                    await set_timeframe(page, timeframe)
                    prev_timeframe = timeframe

                if chart_type != "candle" and chart_type != prev_chart_type:
                    await set_chart_type(page, chart_type)
                prev_chart_type = chart_type

                if chart_type == "renko" and brick_size:
                    await set_renko_brick_size(page, brick_size)

                await asyncio.sleep(3)
                await scroll_to_load_all_bars(page)

                symbol_clean = symbol.replace(":", "_")
                tf = timeframe.upper()
                if chart_type == "renko" and brick_size:
                    fname = f"{symbol_clean}, {tf} renko {brick_size}.csv"
                else:
                    fname = f"{symbol_clean}, {tf}.csv"

                output_path = OUTPUT_DIR / fname
                await export_chart_data(page, output_path)
                succeeded += 1

            except Exception as e:
                print(f"  [Error] Job failed: {e}")
                # If browser died mid-job, abort
                try:
                    await page.evaluate("1")
                except Exception:
                    print("  [Fatal] Browser closed — aborting remaining jobs.")
                    break

            await asyncio.sleep(2)

        print(f"\n{'=' * 60}")
        print(f"Done: {succeeded}/{len(JOBS)} exports succeeded")
        print(f"Files: {OUTPUT_DIR.resolve()}")
        try:
            await browser.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
