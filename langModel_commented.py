import re
from random import randint
import math

# for whenever an unaccounted for word or char occurs
UNK = None
# sentence identifiers
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

# Function to read a file from a given file path
def read_sentences_from_file(file_path):
    with open(file_path, "r") as f:
        return [re.split("\s+", line.rstrip('\n')) for line in f]

# Unigram Model class declaration
class UnigramLanguageModel:
    # Declare Model Initialization
    def __init__(self, sentences, smoothing=False):
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        # Adjust frequency of each word in each sentence by 1
        for sentence in sentences:
            for word in sentence:
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                # adjust corpus length if current word is not a sentence marker
                if ((word != SENTENCE_START) and (word != SENTENCE_END)):
                    self.corpus_length += 1
        # subtract 2 as unigram_frequencies dict contains value for <s> and </s>
        self.unique_words = (len(self.unigram_frequencies) - 2)
        self.smoothing = smoothing

    # Function to calculate word unigram probability
    def calculate_unigram_probability(self, word):
        word_probability_numerator = self.unigram_frequencies.get(word, 0)
        word_probability_denominator = self.corpus_length
        # Checks if smoothing is enabled
        if (self.smoothing):
            word_probability_numerator += 1
            # add one to total unique words for UNK - unknown occurrences
            word_probability_denominator += self.unique_words + 1
        # Return a floating point number representative of the word's probability
        return (float(word_probability_numerator) / float(word_probability_denominator))

    # Function to calculate the probability that a given word occurs in a sentence
    def calculate_sentence_probability(self, sentence, normalize_probability=True):
        sentence_probability_log_sum = 0
        # Adjust probability of each word that is not a sentence marker
        for word in sentence:
            if ((word != SENTENCE_START) and (word != SENTENCE_END)):
                word_probability = self.calculate_unigram_probability(word)
                sentence_probability_log_sum += math.log(word_probability, 2)
        # Return 2^sentence probability sum if we are normalizing, else just return the original sentence prob sum
        return math.pow(2, sentence_probability_log_sum) if normalize_probability else sentence_probability_log_sum

    # Function to initialize, sort, and clean input vocabulary
    def sorted_vocabulary(self):
        full_vocab = list(self.unigram_frequencies.keys())
        full_vocab.remove(SENTENCE_START)
        full_vocab.remove(SENTENCE_END)
        full_vocab.sort()
        full_vocab.append(UNK)
        full_vocab.append(SENTENCE_START)
        full_vocab.append(SENTENCE_END)
        return full_vocab

# Bigram Model class declaration
class BigramLanguageModel(UnigramLanguageModel):
    # Initialization function declaration
    def __init__(self, sentences, smoothing=False):
        UnigramLanguageModel.__init__(self, sentences, smoothing)
        self.bigram_frequencies = dict()
        self.unique_bigrams = set()
        # Adjust frequency of all words in all sentences by 1 unless they are sentence markers
        for sentence in sentences:
            previous_word = None
            for word in sentence:
                if (previous_word != None):
                    self.bigram_frequencies[(previous_word, word)] = self.bigram_frequencies.get((previous_word, word),
                                                                                                 0) + 1
                    # Checks for sentence markers
                    if ((previous_word != SENTENCE_START) and (word != SENTENCE_END)):
                        self.unique_bigrams.add((previous_word, word))
                # Assign the current word to the previous word
                previous_word = word

        # subtract two for the uni model as the unigram_frequencies dict
        # contains a value for <s> and </s>, these need to be included in bigram
        self.unique__bigram_words = len(self.unigram_frequencies)

    # Function to calculate word bigram probability
    def calculate_bigram_probabilty(self, previous_word, word):
        bigram_word_probability_numerator = self.bigram_frequencies.get((previous_word, word), 0)
        bigram_word_probability_denominator = self.unigram_frequencies.get(previous_word, 0)
        # Checks if smoothing is enabled
        if (self.smoothing):
            bigram_word_probability_numerator += 1
            bigram_word_probability_denominator += self.unique__bigram_words
        # Return a floating point number representative of the word's probability or 0 if either num or denom is 0
        return 0.0 if ((bigram_word_probability_numerator == 0) or (bigram_word_probability_denominator == 0)) else (
                    float(
                        bigram_word_probability_numerator) / float(bigram_word_probability_denominator))

    # Function to calculate the probability that a given word occurs in a sentence
    def calculate_bigram_sentence_probability(self, sentence, normalize_probability=True):
        bigram_sentence_probability_log_sum = 0
        previous_word = None
        # Adjust probability of each word that is not a sentence marker
        for word in sentence:
            # Checks if a given word is the first word in its sentence
            if (previous_word != None):
                bigram_word_probability = self.calculate_bigram_probabilty(previous_word, word)
                bigram_sentence_probability_log_sum += math.log(bigram_word_probability, 2)
            previous_word = word
        # Return 2^sentence probability sum if we are normalizing, else just return the original sentence prob sum
        return math.pow(2,
                        bigram_sentence_probability_log_sum) if normalize_probability else bigram_sentence_probability_log_sum


# calculate number of unigrams
def calculate_number_of_unigrams(sentences):
    unigram_count = 0
    # Fix the length of each sentence according to sentence markers
    for sentence in sentences:
        # subtract two for <s> and </s>
        unigram_count += len(sentence) - 2
    # Return the unigram count
    return unigram_count

# calculate number of bigrams
def calculate_number_of_bigrams(sentences):
    bigram_count = 0
    # subtract one for number of bigrams in each sentence
    for sentence in sentences:
        bigram_count += len(sentence) - 1

    return bigram_count

# Function to print the Unigram probabilities
def print_unigram_probs(sorted_vocab_keys, model, word_1):
    # find the right key in the sorted list
    for vocab_key in sorted_vocab_keys:
        # if the key matches, print and calculate the probability, can be <s> or </s>
        if (vocab_key == word_1):
            print("{}: {}".format(vocab_key if vocab_key != UNK else 'UNK',
                                  model.calculate_unigram_probability(vocab_key)), end=' ')

# Function to print the Bigram probabilities
def print_bigram_probs(sorted_vocab_keys, model, word_1, word_2):
    # match both words
    for vocab_key in sorted_vocab_keys:
        # if the key matches and the first word isnt the end of a sentence
        if ((vocab_key != SENTENCE_END) and (vocab_key == word_1)):
            print((word_1 + ', ' + word_2) if vocab_key != UNK else 'UNK', end=': ')

            for vocab_key_second in sorted_vocab_keys:
                # if the second word isnt the start of the sentence and the word matches
                if ((vocab_key_second != SENTENCE_START) and (vocab_key_second == word_2)):
                    print("{0:.5f}".format(model.calculate_bigram_probabilty(vocab_key, vocab_key_second)), end="\t\t")

# Function to find top 10 unigrams
def find_top_ten_unigrams(sorted_vocab_keys, model):
    prob_dict = {}
    top_ten = {}

    # Loop through each key
    for vocab_key in sorted_vocab_keys:
        # find the probability
        prob = (model.calculate_unigram_probability(vocab_key))
        # trying to shorten the dictionary a bit
        if (prob > 0.005):
            prob_dict[vocab_key] = prob

    # Loop through 10 indexes
    for index in range(10):
        # get max key
        top_key = max(prob_dict, key=prob_dict.get)
        # get that keys value
        top_value = prob_dict[top_key]
        # remove the top key while storing it into the top ten list with its probability
        top_ten[top_key] = prob_dict.pop(top_key)
    print('\nTop Ten Unigrams:\n' + str(top_ten))

# Function to find top 10 biigrams
# note that this function takes about a minute
def find_top_ten_bigrams(sorted_vocab_keys, model):
    prob_dict = {}
    top_ten = {}
    # Loop through each key
    for vocab_key in sorted_vocab_keys:
        for vocab_key_second in sorted_vocab_keys:
            # find the probability
            prob = (model.calculate_bigram_probabilty(vocab_key, vocab_key_second))

            # trying to shorten the dictionary a bit
            if (prob > 0.0005):
                prob_dict[str(vocab_key) + ', ' + str(vocab_key_second)] = prob

    # Loop through 10 indexes
    for index in range(10):
        # get max key
        top_key = max(prob_dict, key=prob_dict.get)
        # get that keys value
        top_value = prob_dict[top_key]
        # remove the top key while storing it into the top ten list with its probability
        top_ten[top_key] = prob_dict.pop(top_key)
    print('\nTop Ten Bigrams:\n' + str(top_ten))


# Function to calculate perplexity for unigrams
def calculate_unigram_perplexity(model, sentences):
    # find the number of unigrams
    unigram_count = calculate_number_of_unigrams(sentences)
    sentence_probability_log_sum = 0
    # Loop through each sentence
    for sentence in sentences:
        try:
            # calculate using log to try avoiding underflow and better performance
            sentence_probability_log_sum -= math.log(model.calculate_sentence_probability(sentence), 2)
        # Exception for underflow
        except:
            sentence_probability_log_sum -= float('-inf')

    # finish the calculation, equivalent to math done int class
    return math.pow(2, (sentence_probability_log_sum / unigram_count))

# Function to calculate perplexity for bigrams
def calculate_bigram_perplexity(model, sentences):
    # find the number of unigrams
    number_of_bigrams = calculate_number_of_bigrams(sentences)
    bigram_sentence_probability_log_sum = 0

    # for each sentence
    for sentence in sentences:
        try:
            # calculate using log to try avoiding underflow and better performance
            bigram_sentence_probability_log_sum -= math.log(model.calculate_bigram_sentence_probability(sentence), 2)
        # Exception for underflow
        except:
            bigram_sentence_probability_log_sum -= float('-inf')

    # finish the calculation, equivalent to math done int class
    return math.pow(2, (bigram_sentence_probability_log_sum / number_of_bigrams))


# Function to generate sentences semi-randomly given a set of sorted keys and a Language Model
def generate_sentence(sorted_vocab_keys, model):
    sent = ['<s>']
    nextWord = ''
    probability = 0

    # picks a random starting word for the sentence
    curWord = sorted_vocab_keys[randint(0, len(sorted_vocab_keys))]

    # find next word, caps sentence at 10 and ends at the end sentnce symbol
    while ((len(sent) < 10) and (sent[-1] != '</s>')):
        # go through every possible word
        for vocab_key in sorted_vocab_keys:
            # find the probability of that word given the previous
            tempProb = model.calculate_bigram_probabilty(curWord, vocab_key)

            # find the most likely in the lexicon
            if (probability < tempProb):
                nextWord = vocab_key
                probability = tempProb

        # reset for the new word
        probability = 0
        sent.append(nextWord)
        curWord = nextWord

    print('\nGenerated sentence: \n' + str(sent))
    return sent


# the code should ask for the directory/file name of both these files
# will be commented out for testings sake
dataset = read_sentences_from_file('training.txt')
dataset_test = read_sentences_from_file('test.txt')
dataset = read_sentences_from_file(input('Enter the path to the training text file: '))
dataset_test = read_sentences_from_file(input('Enter the path to the test text file: '))

# given the training and test files, create the models
dataset_model_unsmoothed = BigramLanguageModel(dataset)
dataset_model_smoothed = BigramLanguageModel(dataset, smoothing=True)

# get sorted list of the lexicon
sorted_vocab_keys = dataset_model_unsmoothed.sorted_vocabulary()

# get the words desired by the user for uni/bigram calculations
uni_word = input('Enter your word for unigram: ')
bi_word_1 = input('Enter first bigram word: ')
bi_word_2 = input('Enter second bigram word: ')

# print/calculate unsmoothed and smoothed unigram probability
print('\n=== UNIGRAM MODEL ===\n- Unsmoothed  -')
print_unigram_probs(sorted_vocab_keys, dataset_model_unsmoothed, uni_word)
print('\n\n- Smoothed  -')
print_unigram_probs(sorted_vocab_keys, dataset_model_smoothed, uni_word)

# print/calculate unsmoothed and smoothed bigram probability
print('\n\n=== BIGRAM MODEL ===\n- Unsmoothed  -')
print_bigram_probs(sorted_vocab_keys, dataset_model_unsmoothed, bi_word_1, bi_word_2)
print('\n\n- Smoothed  -')
print_bigram_probs(sorted_vocab_keys, dataset_model_smoothed, bi_word_1, bi_word_2)

# for perplexity
bigram_dataset_model_smoothed = BigramLanguageModel(dataset, smoothing=True)

# print and calculate perplexity for uni/bigrams
print('\n\nPERPLEXITY of test.txt')
print('unigram: ', calculate_unigram_perplexity(bigram_dataset_model_smoothed, dataset_test))
print('bigram: ', calculate_bigram_perplexity(bigram_dataset_model_smoothed, dataset_test))

# what will call and print generated sentences
generate_sentence(sorted_vocab_keys, dataset_model_smoothed)
generate_sentence(sorted_vocab_keys, dataset_model_smoothed)
generate_sentence(sorted_vocab_keys, dataset_model_smoothed)

# what will find and print the top ten uni/bigrams with their probabilities
find_top_ten_unigrams(sorted_vocab_keys, dataset_model_smoothed)
find_top_ten_bigrams(sorted_vocab_keys, dataset_model_smoothed)