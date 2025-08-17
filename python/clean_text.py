from colorama import Fore 
import unicodedata
import re

# Clean text to make it optimal for content generation using a LLM model

class CleanText:
    def __init__(self, text, unique_non_english_characters=None):
        self.text = text 
        self.words = text.split()
        self.unique_non_english_characters = unique_non_english_characters
        
    def contains_private_unicode(self, char: str) -> bool:
        """Check if a character falls within the Unicode Private Use Area."""
        return '\uE000' <= char <= '\uF8FF'

    def find_words_with_private_unicode(self):
        """Find words containing any private-use Unicode characters."""
        return [word for word in self.words if any(self.contains_private_unicode(c) for c in word)]

    def find_words_with_non_ascii(self):
        """Find words containing any non-ASCII characters (no english chars)."""
        return [word for word in self.words if any(ord(c) > 127 for c in word)]

    def extract_non_ascii_characters(self):
        """Extract all non-ASCII characters from the text."""
        return {char for char in self.text if ord(char) > 127}

    #Normalize text for consistent Unicode handling
    #NFKC helps convert compatible characters into standard ones, e.g., ’ → ', ﬁ → fi. (diﬀers word)
    def normalize_text(self):
        return unicodedata.normalize("NFKC", self.text)

    def replace_smart_and_special_chars(self, text):
        """Replace common smart quotes, symbols, and dashes with ASCII equivalents."""
        replacements = {
            "–": "-", "—": "-", "—": "-", "»": "", "/": " ", "…": "...", "×": "x",
            "ü": "u", "ø": "o", "ł": "l", "®": " ", "©": " ", "\u2022": " ", "\u00D8": "o",
            "\u00A9": " ", "’": "'", "‘": "'", '“': '"', '”': '"',
            "\u00E9": "e", "\u00EB": "e"
        }
        for original, replacement in replacements.items():
            text = text.replace(original, replacement)
        return text

    def remove_number_ranges(self, text):
        """Remove number ranges like '100–200' using dash or Unicode dash."""
        return re.sub(r'\b\d+[\u2013\u2014\u2012\u2010\u2212-]\d+\.*', '', text)
    

    # def replace_non_ascii_chars(self, word):
    #     """Replace bad glyphs or unusual non-ASCII characters in a word."""

    #     # Normalize the word to decompose combining characters
    #     word = unicodedata.normalize("NFKC", word)

    #     replacements = {
    #         "\u0346": "fl",  
    #         "\u0345": "fi",
    #         "\u00EF": "i"
    #     }

    #     if self.unique_non_english_characters is None:
    #         self.unique_non_english_characters = set(replacements.keys())

    #     for char in replacements:
    #         if char in word:
    #             word = word.replace(char, replacements[char])

    #     return word

    def replace_non_ascii_chars(self, word):
        """Replace bad glyphs or unusual non-ASCII characters in a word."""

        # Normalize (compatibility decomposition: splits ligatures/combining marks)
        #word = unicodedata.normalize("NFKC", word)

        replacements = {
            # Common ligatures
            "\uFB01": "fi",     # ﬁ
            "\uFB02": "fl",     # ﬂ
            "\uFB03": "ffi",    # ﬃ
            "\uFB04": "ffl",    # ﬄ
            "\uFB00": "ff",     # ﬀ
            "\uFB05": "ft",     # ﬅ
            "\uFB06": "st",     # ﬆ
            "\u0346": "fl",  
            "\u0345": "fi",
            "\u00EF": "i",

            # Accented/diacritic variants
            "\u00EF": "i",      # ï
            "\u00EE": "i",      # î
            "\u00ED": "i",      # í
            "\u00EC": "i",      # ì
            "\u00E1": "a",      # á
            "\u00E0": "a",      # à
            "\u00E2": "a",      # â
            "\u00E4": "a",      # ä
            "\u00E3": "a",      # ã
            "\u00E5": "a",      # å
            "\u00E7": "c",      # ç
            "\u00E9": "e",      # é
            "\u00E8": "e",      # è
            "\u00EA": "e",      # ê
            "\u00EB": "e",      # ë
            "\u00F1": "n",      # ñ
            "\u00F3": "o",      # ó
            "\u00F2": "o",      # ò
            "\u00F4": "o",      # ô
            "\u00F6": "o",      # ö
            "\u00F5": "o",      # õ
            "\u00FA": "u",      # ú
            "\u00F9": "u",      # ù
            "\u00FB": "u",      # û
            "\u00FC": "u",      # ü
            "\u00FD": "y",      # ý
            "\u00FF": "y",      # ÿ

            # Dashes and hyphens
            "\u2013": "-",      # –
            "\u2014": "-",      # —
            "\u2212": "-",      # −
            "\u2012": "-",      # ‒

            # Quotes
            "\u201C": "\"",     # “
            "\u201D": "\"",     # ”
            "\u201E": "\"",     # „
            "\u201F": "\"",     # ‟
            "\u2018": "'",      # ‘
            "\u2019": "'",      # ’
            "\u201A": "'",      # ‚
            "\u201B": "'",      # ‛

            # Ellipsis
            "\u2026": "...",    # …

            # Spaces
            "\u00A0": " ",      # non-breaking space
            "\u2002": " ",      # en space
            "\u2003": " ",      # em space
            "\u2009": " ",      # thin space
            "\u202F": " ",      # narrow no-break space
            "\u3000": " ",      # ideographic space
            "\u200B": "",       # zero-width space
            "\u200C": "",       # zero-width non-joiner
            "\u200D": "",       # zero-width joiner

            # Misc
            "\u2022": "-",      # • bullet
            "\u00B7": "-",      # · middle dot
            "\u2219": "-",      # ∙ dot operator
            "\u25CB": "o",      # ○
            "\u00B0": " degrees", # °
            "\u00BC": "1/4",    # ¼
            "\u00BD": "1/2",    # ½
            "\u00BE": "3/4",    # ¾
        }

        for bad, good in replacements.items():
            word_og = word  # Keep original for debugging
            word = word.replace(bad, good)
            if word != word_og:
                print(Fore.RED + f"Replaced '{bad}' with '{good}' in word: {word_og} -> {word}" + Fore.RESET)


        return word 

    def clean_word(self, word):
        """Apply all cleaning steps to an individual word."""
        word = self.replace_smart_and_special_chars(word)
        word = unicodedata.normalize("NFKC", word)
        word = self.remove_number_ranges(word)
        word = self.replace_non_ascii_chars(word)
        return word

    def clean_all(self):
        """Clean the full text and return cleaned version."""
        text = self.replace_smart_and_special_chars(self.text)
        text = unicodedata.normalize("NFKC", text)
        text = self.remove_number_ranges(text)
        cleaned_words = [self.replace_non_ascii_chars(word) for word in text.split()]
        return ' '.join(cleaned_words)