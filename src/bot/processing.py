import os
import logging


from bot.inference import InferenceModel
from bot.utils import save_image, DirsInstance


logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


async def process_img(context, chat_id, client_id) -> None:
    usr_data_dir = DirsInstance().data_dir / client_id
    usr_res_dir = DirsInstance().results_dir / client_id
    usr_cache_real_dir = DirsInstance().cache_real / client_id
    usr_cache_fake_dir = DirsInstance().cache_fake / client_id

    for img_name in os.listdir(usr_data_dir):
        res_img = await InferenceModel().process(img_path=usr_data_dir / img_name)
        save_image(res_img, usr_res_dir / img_name)
        await context.bot.send_message(
            chat_id=chat_id, text="Image processed! Sending result..."
        )
    await send_images(context, client_id, chat_id)
    for img_name in os.listdir(usr_data_dir):
        os.replace(usr_data_dir / img_name, usr_cache_real_dir / img_name)
        os.replace(usr_res_dir / img_name, usr_cache_fake_dir / img_name)


async def send_images(context, client_id, chat_id) -> None:
    text = "The result:"
    await context.bot.send_message(chat_id=chat_id, text=text)

    for image_name in os.listdir(DirsInstance().results_dir / client_id):
        with open(DirsInstance().results_dir / client_id / image_name, "rb") as image:
            await context.bot.send_photo(chat_id=chat_id, photo=image)
