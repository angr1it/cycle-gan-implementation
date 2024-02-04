import logging

from telegram import Update
from telegram.ext import CallbackContext

from bot.processing import process_img
from bot.utils import create_dirs, DirsInstance


logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Send a photo you need to translate to monet-style:")


async def help_command(update: Update, context: CallbackContext) -> None:
    text = "This bot applies monet-styles to the image sent."
    await update.message.reply_text(text)


async def get_photo(update: Update, context: CallbackContext) -> None:

    file = await update.message.effective_attachment[-1].get_file()
    client_id = update.message.chat.username
    message_id = update.message.message_id
    file_name = f"{client_id}_{message_id}.png"

    logging.info(f"get_photo called for {client_id}")
    create_dirs(client_id)
    await file.download_to_drive(DirsInstance().data_dir / client_id / file_name)

    text = "processing..."
    await update.message.reply_text(text)
    await process_img(context, update.effective_chat.id, client_id)
