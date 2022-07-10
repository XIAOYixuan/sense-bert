import json
import copy
import os


class InputExample(object):

    def __init__(self, guid:str, text_a:str, text_b:str, tgt_word:str, \
        word_a:str, word_a_pos:int, word_a_char_pos:int, \
        word_b:str, word_b_pos:int, word_b_char_pos:int, \
        label:int):
        assert word_a in text_a.split()[word_a_pos] 
        assert word_b in text_b.split()[word_b_pos] 
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.tgt_word = tgt_word
        self.word_a = word_a
        self.word_a_pos = word_a_pos
        self.word_a_char_pos = word_a_char_pos
        self.word_b = word_b
        self.word_b_pos = word_b_pos
        self.word_b_char_pos = word_b_char_pos
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class WicFewShotLoader:

    def __init__(self, path):
        self.path = path
        self.train = self._load_file("train")
        self.valid = self._load_file("dev32")
        self.test = self._load_file("val")


    def _load_file(self, set_type: str):
        examples = []
        set_type_file = set_type + '.jsonl'
        path = os.path.join(self.path, set_type_file)
        with open(path, encoding='utf8') as fp:
            for line in fp:
                example_json = json.loads(line)
                word_a, a_pos, a_char_pos = self._get_surface(1, example_json)
                word_b, b_pos, b_char_pos = self._get_surface(2, example_json)
                sent_a = example_json['sentence1']
                sent_b = example_json['sentence2']
                tgt_word = example_json['word']
                guid = f"{set_type}-{example_json['idx']}"
                label = 1 if example_json.get('label') else 0
                example = InputExample(guid=guid, text_a=sent_a, text_b=sent_b, \
                    tgt_word=tgt_word, \
                    word_a=word_a, word_a_pos=a_pos, word_a_char_pos=a_char_pos,\
                    word_b=word_b, word_b_pos=b_pos, word_b_char_pos=b_char_pos, \
                    label=label)
                examples.append(example)
        return examples


    def _get_surface(self, sent_id:int, example_json):
        sent_id = str(sent_id)
        text = example_json['sentence'+sent_id]
        st, ed = example_json['start'+sent_id], example_json['end'+sent_id]
        token_position = self._get_token_position(text, st, ed)
        tgt_word = example_json['sentence'+sent_id][st:ed]
        # print(text, token_position)
        # print(text.split()[token_position], tgt_word)
        return tgt_word, token_position, (st, ed)


    def _get_token_position(self, text, st, ed):
        if st == 0:
            return 0
        cnt = 1
        for idx in range(len(text)):
            if idx == st:
                break
            if text[idx]  == ' ':
                cnt += 1
        return cnt-1