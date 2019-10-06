import json
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer


def main():
    print("Parse started")
    with open('master&margarita.txt', 'r', encoding='utf-8') as content_file:
        text = content_file.read()

        splitted_chunks = list(filter(None, text.split('\n')))

        sentences = []

        for chunk in splitted_chunks:
            sentences.extend(extract_sentences(chunk))

        save_sentences(sentences)


def extract_sentences(text):
    sentence_list = sent_tokenize(text)
    return sentence_list


def save_sentences(sentences):
    current_id = 1

    with open("phrases.json", "w", encoding='utf-8') as out_file:
        sentences_json = []
        word_tokenizer = RegexpTokenizer(r'\w+')

        for phrase in sentences:

            word_count = len(word_tokenizer.tokenize(phrase))

            if len(phrase) <= 3 or word_count == 1 or word_count > 20:
                continue

            json_phrase = {
                'id': str(current_id),
                'phrase': str(phrase)
            }
            sentences_json.append(json_phrase)

            current_id += 1

        json.dump(sentences_json, out_file, ensure_ascii=False)


if __name__ == '__main__':
    main()
