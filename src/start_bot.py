import os
import logging

from dotenv import load_dotenv

from telegram.ext import (
    Application,
    filters,
    MessageHandler,
    CommandHandler,
)

from bot.commands import start, help_command, get_photo


load_dotenv()
TELEGRAM_TOKEN_API = os.environ.get("TELEGRAM_TOKEN_API")


def main():

    application = Application.builder().token(TELEGRAM_TOKEN_API).build()
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, get_photo))
    application.run_polling()

    logging.log("Bot started.")


if __name__ == "__main__":
    main()
