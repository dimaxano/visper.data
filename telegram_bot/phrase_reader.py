import json
from random import randint

from utils import get_stats_dict, put_stats_dict, PHRASES_FILE_PATH


class PhraseReader:
    def __init__(self, file_name):
        input_file = open(file_name, encoding='utf-8')
        self.json_array = json.load(input_file)

    def get_phrase(self, index):
        return self.json_array[index]['phrase']
    
    def get_next_phrase(self, username):
        """
            Returns the next phrase for the user
        """
        
        stats_dict = get_stats_dict()
        phrase_reader = PhraseReader(PHRASES_FILE_PATH)
        
        if username in stats_dict:
            stats_dict[username] += 1
        else:
            stats_dict[username] = 0
        put_stats_dict(stats_dict)

        next_index = stats_dict[username] + 1
        next_phrase = phrase_reader.get_phrase(next_index)
        return next_phrase
    
    def get_current_idx(self, username):
        stats_dict = get_stats_dict()

        return stats_dict[username]