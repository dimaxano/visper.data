import os
from os.path import join
from datetime import datetime

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging

from phrase_reader import PhraseReader
from utils import DATA_DIR_PATH, WELCOME_TEXT, PHRASES_FILE_PATH

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

phrase_reader = PhraseReader(PHRASES_FILE_PATH)

# Define a few command handlers. These usually take the two arguments bot and
# update. Error handlers also receive the raised TelegramError object in error.
def start(bot, update):
    """Send a message when the command /start is issued."""
    update.message.reply_text(WELCOME_TEXT)

    username = update.effective_user.full_name
    phrase = phrase_reader.get_next_phrase(username)
    update.message.reply_text(phrase)


def help(bot, update):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def process_message(bot, update):
    username = update.effective_user.full_name
    video = update.message.video

    save_dir = join(DATA_DIR_PATH, username)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if video is not None:
        datetime_str = datetime.now().timestamp()
        phrase_idx = phrase_reader.get_current_idx(username)

        file_name = f"{datetime_str}_{phrase_idx:04}.mp4"
        file_path = join(save_dir, file_name)

        video_file = bot.get_file(video.file_id)
        video_file.download(custom_path=file_path)

    phrase = phrase_reader.get_next_phrase(username)
    update.message.reply_text(phrase)


def repeat_text(bot, update):
    username = update.effective_user.full_name

    phrase = phrase_reader.get_next_phrase(username)
    update.message.reply_text(phrase)


def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():

    """Start the bot."""
    # Create the EventHandler and pass it your bot's token.
    tg_bot_token = ""
    updater = Updater(token=tg_bot_token)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(CommandHandler("text", repeat_text))

    dp.add_handler(MessageHandler(Filters.video, process_message))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling(timeout=25)

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
