import csv
import os
import pandas as pd

from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

# def create_song_language_dict(file_path):
#     df = pd.read_csv(file_path)

#     def get_song_name(filepath):
#         filename_without_ext = os.path.splitext(os.path.basename(filepath))[0]
#         return filename_without_ext

#     df['song_name'] = df['Filepath'].apply(get_song_name)
#     song_language_dict = df.dropna(subset=['song_name']).set_index('song_name')['Language'].to_dict()

#     return song_language_dict

# file_path = 'path/to/JamendoLyrics.csv'
# lang_dict = create_song_language_dict(file_path)

lang_dict = {"Avercage_-_Embers": "English"}

# Prepare phonemizer
separator = Separator(phone=';', word=None)
punct = Punctuation(';:,.!"?()-')

langs = ['English', 'French', 'German', 'Italian', 'Spanish']
espeak_ids = {
    'English': 'en-us',
    'French': 'fr-fr',
    'German': 'de',
    'Italian': 'it',
    'Spanish': 'es'
}

# Process each language and song
for lang in langs:
    # Initialize backend for the current language
    backend = EspeakBackend(espeak_ids[lang])
    
    for song, language in lang_dict.items():
        if language == lang:
            file_path = os.path.join("./jamendo_example/annot/", song + ".words.txt")
            out_path = os.path.join("./jamendo_example/annot/", song + ".csv")
            
            # Read the CSV file into a list of dictionaries
            data_to_write = []
            st = 0
            ed = st
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    word = line.strip()
                    if word:
                        cleaned = punct.remove(word).lower()
                        phone = backend.phonemize([cleaned], separator=separator, strip=True)[0]
                        ed += len(phone.split(";"))
                        # Append the word and its phonemes to our list
                        data_to_write.append([word, phone, [st, ed]])
                        st = ed + 1
                        ed = st
            
            # Write the modified data back to the CSV file
            with open(out_path, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(['word', 'phonemizer', 'phone_idx'])
                writer.writerows(data_to_write)