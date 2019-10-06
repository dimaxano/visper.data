import csv
import json



DATA_DIR_PATH = "uploads/"
WELCOME_TEXT = ("Привет! Меня зовут Visper.\nСледующим сообщением я отправил тебе предложение. " 
                "Произнеси его, записывая себя на фронтальную камеру, и отправь мне.\n"
                "Старайся, чтобы в начале и в конце видео, не было длинных пауз. ☝️\n"
                "Если возникнут вопросы, отправь мне /help.\nСпасибо 😉!")
PHRASES_FILE_PATH = "phrases.json"
USER_STATS_PATH = "user_stats.csv"


def get_stats_dict():
    """
        Method return user stats dict
    """

    stats_dict = dict()
    with open(USER_STATS_PATH, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            if line:
                stats_dict[line[0]] = int(line[1])

    return stats_dict


def put_stats_dict(stats_dict):
    """
        Method fot writing in user stats file
    """
    with open(USER_STATS_PATH, 'w', newline='') as csvfile:
        stats_writer = csv.writer(csvfile, delimiter=",")

        for username in stats_dict:
            row = [username, stats_dict[username]]
            stats_writer.writerow(row)
    

def get_current_idx(username):
    stats_dict = get_stats_dict()

    return stats_dict[username]
