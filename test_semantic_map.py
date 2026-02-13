"""Browser test for Semantic Map feature at http://127.0.0.1:8001"""
import asyncio
import sys

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Installing playwright...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    from playwright.async_api import async_playwright


async def main():
    console_errors = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        
        page = await context.new_page()
        
        # Capture console errors
        def on_console(msg):
            if msg.type in ("error", "warning"):
                console_errors.append({"type": msg.type, "text": msg.text})
        page.on("console", on_console)
        
        try:
            # Step 1: Navigate and look at the page
            print("Step 1: Navigating to http://127.0.0.1:8001 ...")
            await page.goto("http://127.0.0.1:8001", wait_until="networkidle", timeout=15000)
            
            # Check for Home and Semantic tabs
            home_tab = await page.query_selector('[data-main-tab="home"]')
            semantic_tab = await page.query_selector('[data-main-tab="semantic"]')
            print(f"  Home tab found: {home_tab is not None}")
            print(f"  Semantic tab found: {semantic_tab is not None}")
            
            # Step 2: Click Semantic tab
            print("\nStep 2: Clicking Semantic tab...")
            await semantic_tab.click()
            await page.wait_for_timeout(500)
            
            # Step 3: Take screenshot and describe
            print("\nStep 3: Initial state after clicking Semantic...")
            await page.screenshot(path="d:/bot/semantic_step3.png")
            
            loading = await page.query_selector("#smap-loading")
            empty = await page.query_selector("#smap-empty")
            plot = await page.query_selector("#smap-plot")
            loading_msg = await page.text_content("#smap-loading-msg")
            
            loading_visible = await loading.evaluate("el => !el.classList.contains('hidden')") if loading else False
            empty_visible = await empty.evaluate("el => !el.classList.contains('hidden')") if empty else False
            
            print(f"  Loading overlay visible: {loading_visible}")
            print(f"  Empty state visible: {empty_visible}")
            print(f"  Loading message: {loading_msg}")
            print(f"  Plot container exists: {plot is not None}")
            
            # Step 4: Wait 15 seconds
            print("\nStep 4: Waiting 15 seconds for computation...")
            await page.wait_for_timeout(15000)
            
            # Take second screenshot
            await page.screenshot(path="d:/bot/semantic_step4.png")
            
            loading = await page.query_selector("#smap-loading")
            loading_visible = await loading.evaluate("el => !el.classList.contains('hidden')") if loading else False
            loading_msg = await page.text_content("#smap-loading-msg")
            paper_count = await page.text_content("#smap-paper-count")
            
            print(f"  Loading overlay visible: {loading_visible}")
            print(f"  Loading message: {loading_msg}")
            print(f"  Paper count: {paper_count}")
            
            # Step 5: If plot visible, add keyword filter
            if not loading_visible:
                print("\nStep 5: Adding keyword filter...")
                keyword_input = await page.query_selector("#keyword-input")
                if keyword_input:
                    await keyword_input.fill("machine")
                    await keyword_input.press("Enter")
                    await page.wait_for_timeout(1000)
                    await page.screenshot(path="d:/bot/semantic_step5.png")
                    print("  Keyword 'machine' added, screenshot saved.")
                else:
                    print("  Keyword input not found.")
            else:
                print("\nStep 5: Skipped - plot not visible (still loading or error)")
            
            # Step 6: Report console errors
            print("\nStep 6: JavaScript console errors:")
            for err in console_errors:
                print(f"  [{err['type']}] {err['text']}")
            if not console_errors:
                print("  (none captured)")
                
        except Exception as e:
            print(f"\nError: {e}")
            await page.screenshot(path="d:/bot/semantic_error.png")
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
