import pickle
import os


class OfflineDict:

    def __init__(self, offline_dict_path='./offline_dict.pkl'):
        self.offline_dict_path = offline_dict_path
        if os.path.exists(offline_dict_path):
            with open(offline_dict_path, 'rb') as file:
                self.offline_dict = pickle.load(file)
        else:
            self.offline_dict = {}
            self.save()

    def get_value(self, key):
        return self.offline_dict.get(key, None)

    def add_entry(self, key, value):
        self.offline_dict[key] = value
        self.save()

    def save(self):
        with open(self.offline_dict_path, 'wb') as file:
            pickle.dump(self.offline_dict, file)


chars_to_del = ['.', ',']

chars_to_del = set(chars_to_del)


filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']

numbers = [
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five",
    "twenty-six", "twenty-seven", "twenty-eight", "twenty-nine", "thirty",
    "thirty-one", "thirty-two", "thirty-three", "thirty-four", "thirty-five",
    "thirty-six", "thirty-seven", "thirty-eight", "thirty-nine", "forty",
    "forty-one", "forty-two", "forty-three", "forty-four", "forty-five",
    "forty-six", "forty-seven", "forty-eight", "forty-nine", "fifty",
    "fifty-one", "fifty-two", "fifty-three", "fifty-four", "fifty-five",
    "fifty-six", "fifty-seven", "fifty-eight", "fifty-nine", "sixty",
    "sixty-one", "sixty-two", "sixty-three", "sixty-four", "sixty-five",
    "sixty-six", "sixty-seven", "sixty-eight", "sixty-nine", "seventy",
    "seventy-one", "seventy-two", "seventy-three", "seventy-four", "seventy-five",
    "seventy-six", "seventy-seven", "seventy-eight", "seventy-nine", "eighty",
    "eighty-one", "eighty-two", "eighty-three", "eighty-four", "eighty-five",
    "eighty-six", "eighty-seven", "eighty-eight", "eighty-nine", "ninety",
    "ninety-one", "ninety-two", "ninety-three", "ninety-four", "ninety-five",
    "ninety-six", "ninety-seven", "ninety-eight", "ninety-nine", "one hundred"
]

ordinals = [
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
    "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
    "twenty-first", "twenty-second", "twenty-third", "twenty-fourth", "twenty-fifth",
    "twenty-sixth", "twenty-seventh", "twenty-eighth", "twenty-ninth", "thirtieth",
    "thirty-first", "thirty-second", "thirty-third", "thirty-fourth", "thirty-fifth",
    "thirty-sixth", "thirty-seventh", "thirty-eighth", "thirty-ninth", "fortieth",
    "forty-first", "forty-second", "forty-third", "forty-fourth", "forty-fifth",
    "forty-sixth", "forty-seventh", "forty-eighth", "forty-ninth", "fiftieth",
    "fifty-first", "fifty-second", "fifty-third", "fifty-fourth", "fifty-fifth",
    "fifty-sixth", "fifty-seventh", "fifty-eighth", "fifty-ninth", "sixtieth",
    "sixty-first", "sixty-second", "sixty-third", "sixty-fourth", "sixty-fifth",
    "sixty-sixth", "sixty-seventh", "sixty-eighth", "sixty-ninth", "seventieth",
    "seventy-first", "seventy-second", "seventy-third", "seventy-fourth", "seventy-fifth",
    "seventy-sixth", "seventy-seventh", "seventy-eighth", "seventy-ninth", "eightieth",
    "eighty-first", "eighty-second", "eighty-third", "eighty-fourth", "eighty-fifth",
    "eighty-sixth", "eighty-seventh", "eighty-eighth", "eighty-ninth", "ninetieth",
    "ninety-first", "ninety-second", "ninety-third", "ninety-fourth", "ninety-fifth",
    "ninety-sixth", "ninety-seventh", "ninety-eighth", "ninety-ninth", "one hundredth"
]

filter_words = set(filter_words + numbers + ordinals)