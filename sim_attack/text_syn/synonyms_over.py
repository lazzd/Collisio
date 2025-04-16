import subprocess
import sys

import nltk
from nltk.corpus import wordnet as wn
import spacy

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Sized, Tuple, Union
import logging
import traceback

from wordhoard import Synonyms


logger = logging.getLogger(__name__)


def check_and_install_spacy_model():
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("SpaCy model 'en_core_web_sm' is not installed. Download...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load('en_core_web_sm')
    return nlp


nlp = check_and_install_spacy_model()
#nlp = spacy.load("en_core_web_sm")


# { Part-of-speech constants
ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"
# }

POS_LIST = [NOUN, VERB, ADJ, ADV]

POS_DICT = {
    NOUN: "noun",
    VERB: "verb",
    ADJ: "adj",
    ADV: "adv"
}


def colorized_text(text: str, color: str) -> None:
    """
    This function provides terminal error messages in color.
    :param text: text to colorized
    :param color: color value
    :return: print error in color
    """
    RESET = "\033[0m" # resets the terminal color to its default
    if color == 'red':
        print(f'\033[1;31m{text}{RESET}')
    elif color == 'blue':
        print(f'\033[1;34m{text}{RESET}')
    elif color == 'green':
        print(f'\033[1;32m{text}{RESET}')
    elif color == 'magenta':
        print(f'\033[1;35m{text}{RESET}')


class Synonyms_self_wordnet(Synonyms):
    def __init__(self,
                search_string: str = '',
                output_format: str = 'list',
                max_number_of_requests: int = 30,
                rate_limit_timeout_period: int = 60,
                user_agent: Optional[str] = None,
                proxies: Optional[Dict[str, str]] = None,
                pos_local_wordnet: str = NOUN):
        super().__init__(search_string, output_format, max_number_of_requests, rate_limit_timeout_period, user_agent, proxies)
        self.pos_local_wordnet = pos_local_wordnet
    
    def _backoff_handler(self, details):
        if self._rate_limit_status is False:
            colorized_text('The synonyms query rate limit was reached. The querying process is '
                                 'entering a temporary hibernation mode.', 'red')
            logger.info('The synonyms query rate limit was reached.')
            self._rate_limit_status = True

    def _run_query_tasks_in_parallel(self) -> List[tuple[List[str], str]]:
        """
        Runs the query tasks in parallel using a ThreadPool.

        :return: list
        :rtype: nested list
        """
        tasks = [self._query_collins_dictionary, self._query_merriam_webster, self._query_synonym_com,
                 self._query_thesaurus_com, self._local_query_wordnet]

        with ThreadPoolExecutor(max_workers=5) as executor:
            running_tasks = []
            finished_tasks = []
            try:
                for task in tasks:
                    submitted_task = executor.submit(task)
                    running_tasks.append(submitted_task)
                for finished_task in as_completed(running_tasks):
                    finished_tasks.append(finished_task.result())
                return finished_tasks
            except Exception as error:
                logger.error('An unknown error occurred in the following code segment:')
                logger.error(''.join(traceback.format_tb(error.__traceback__)))
    
    def _local_query_wordnet(self) -> Union[Tuple[List[str], str], None]:
        """
        This function queries local wordnet for synonyms associated
        with the specific word provided to the Class Synonyms.

        :returns:
            synonyms: list of synonyms

        :rtype: list
        """
        synsets = wn.synsets(self._word, self.pos_local_wordnet)

        norm_word = self._word.lower().replace('_', ' ')

        synonyms_list = []

        for synset in synsets:
            syn_list = synset.lemma_names()
            for syn in syn_list:
                syn = syn.lower().replace('_', ' ')
                if syn != norm_word and syn not in synonyms_list:
                    synonyms_list.append(syn)

        if len(synonyms_list) == 0:
            return None
        else:
            return (synonyms_list, POS_DICT[self.pos_local_wordnet])


class Conj_Synonyms():
    def __init__(self, conjugate=True):
        self.conjugate = conjugate
    
    def _get_pos(self, word):
        pos = nltk.pos_tag([word])[0][1][0].upper()
        if pos in ['J', 'N', 'V', 'R']:
            return pos
        else:
            return None

    def _get_lemma(self, word):
        doc = nlp(word)
        for token in doc:
            return token.lemma_

    def _conjugate_verb(self, original_word, base_form):
        doc = nlp(original_word)
        token = doc[0]
        if token.tag_ == 'VBZ':  # third person singular present
            return base_form + 's'
        elif token.tag_ == 'VBD':  # past tense
            return nltk.stem.SnowballStemmer("english").stem(base_form) + 'ed'
        elif token.tag_ == 'VBG':  # present participle
            return base_form + 'ing'
        elif token.tag_ == 'VBN':  # past participle
            return nltk.stem.SnowballStemmer("english").stem(base_form) + 'ed'
        else:
            return base_form

    def _conjugate_word(self, base_form, original_word):
        doc = nlp(original_word)
        for token in doc:
            if token.tag_.startswith('V'):  # Verb
                return self._conjugate_verb(original_word, base_form)
            elif token.tag_ == 'NNS':  # Plural noun
                return base_form + 's'
            elif token.tag_ == 'NN':  # Singular noun
                return base_form
            else:
                return base_form

    def find_synonyms(self, word, max_number_of_requests: int = 30, rate_limit_timeout_period: int = 60):
        if self.conjugate:
            base_form = self._get_lemma(word)
            # print("BASE", base_form)
            synonym = Synonyms_self_wordnet(search_string=base_form, max_number_of_requests=max_number_of_requests, rate_limit_timeout_period=rate_limit_timeout_period)
            synonyms = synonym.find_synonyms()
            if synonyms is None:
                synonyms = []
            substitutes = [self._conjugate_word(synonym, word) for synonym in synonyms]
            return substitutes
        else:
            synonym = Synonyms_self_wordnet(search_string=word, max_number_of_requests=max_number_of_requests, rate_limit_timeout_period=rate_limit_timeout_period)
            substitutes = synonym.find_synonyms()
            if substitutes is None:
                substitutes = []
            return substitutes