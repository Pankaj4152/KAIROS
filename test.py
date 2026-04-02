import asyncio
from dotenv import load_dotenv
from runtime.tools.messaging import send_message

async def test():
    # Make sure .env is loaded to pick up TELEGRAM_BOT_TOKEN and TELEGRAM_USER_ID
    load_dotenv()
    
    # Test valid message
    print("\nSending normal message...")
    result1 = await send_message("Hello! This is a proactive test.")
    print("Result 1:", result1)
    
    # Test length limit rejection
    print("\nSending 5000 chars...")
    long_msg = "A" * 5000
    result2 = await send_message(long_msg)
    print("Result 2:", result2)

if __name__ == "__main__":
    asyncio.run(test())
