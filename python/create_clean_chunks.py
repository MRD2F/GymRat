import unicodedata
import re
#from transformers import GPT2TokenizerFast
#from langchain.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Read a .pdf file, and splits it into chunks. The chunks are processed to clean them to create an
#optimal input for content generation using a LLM model

class CleanText:
    def __init__(self, text, unique_non_english_characters=None):
        self.text = text #chuck.page_content
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
    #     replacements = {
    #         "\u0346": "fl",
    #         "\u0345": "fi",
    #         "\u00EF": "i"
    #     }
    #     for char in word:
    #         if char in self.unique_non_english_characters and char in replacements:
    #             word = word.replace(char, replacements[char])
    #     return word
    

    def replace_non_ascii_chars(self, word):
        """Replace bad glyphs or unusual non-ASCII characters in a word."""

        # Normalize the word to decompose combining characters
        word = unicodedata.normalize("NFKC", word)

        replacements = {
            "\u0346": "fl",  
            "\u0345": "fi",
            "\u00EF": "i"
        }

        if self.unique_non_english_characters is None:
            self.unique_non_english_characters = set(replacements.keys())

        for char in replacements:
            if char in word:
                word = word.replace(char, replacements[char])

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
    #chunks -> chunk.page_content == text -> words (chunk[i].split()/text.split()) -> word = text.split()[i] 


#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################


class CreateChunks:
    def __init__(self, file_path, chunk_size=500, chunk_overlap=50, length_function=len, return_as_documents=True):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.return_as_documents = return_as_documents
        self.chunks = self.get_chunks()
        self.non_ascii_words, self.private_unicode_words, self.non_ascii_chars  = self.get_problematic_lists()

    def document_loader(self):
        loader = PyMuPDFLoader(self.file_path)
        loaded_document = loader.load()
        return loaded_document

    def split_text(self, document):  

        separators = ["\n\n", "\n", ".", " ", ""]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function, 
            separators=separators
        )
        if self.return_as_documents:
            return text_splitter.split_documents(document)
        else:
            return text_splitter.split_text(document)
        
    def get_chunks(self):
        document = self.document_loader()
        chunks = self.split_text(document)
        return chunks
    
    def get_problematic_lists(self):
        # Create lists with problematic words or characters for all the chunks in the provided file
        # Used later for the CleanText class to remove or fix this words. Also used manually for debbuging and checking the quality of the resulting cleaned chunks 
        non_ascii_words, private_unicode_words, non_ascii_chars = set(), set(), set()
        for chunk in self.chunks:
            #Normalize text for consistent Unicode handling
            text = unicodedata.normalize("NFKC", chunk.page_content)

            clean_text = CleanText(text)

            non_ascii_words.update(clean_text.find_words_with_non_ascii())
            private_unicode_words.update(clean_text.find_words_with_private_unicode())
            non_ascii_chars.update(clean_text.extract_non_ascii_characters())

        non_ascii_words = list(sorted(non_ascii_words))
        private_unicode_words = list(sorted(private_unicode_words))
        non_ascii_chars = list(sorted(non_ascii_chars))

        #print(len(non_ascii_words), len(private_unicode_words), len(non_ascii_chars))
        return non_ascii_words, private_unicode_words, non_ascii_chars

    def get_clean_text(self, text):
        return CleanText(text, self.non_ascii_words)

    def clean_chunk(self, text):
        clean_text = self.get_clean_text(text)
        words = text.split() # chunk == text
        clean_chunk = ""
        for word in words:
            new_text =  clean_text.clean_word(word)
            if new_text == None:
                print(word, "    NONE \\n ")
            clean_chunk += new_text + " "
        return clean_chunk.strip()

    def get_clean_chunks(self):
        text_chunks=[]
        n=0
        #chunks = self.get_chunks()
        for chunk in self.chunks:
            text_chunk = chunk.page_content 
            print(f"Processing chuck number: {n}")
            text_chunk = self.clean_chunk(text_chunk)
            text_chunks.append(text_chunk)
            n+=1
        return text_chunks

test=False
if test:
    file="../data/books/Rebuilding Milo The Lifters Guide to Fixing Common Injuries and Building a Strong Foundation for Enhancing Performance (Dr. Aaron Horschig, Kevin Sonthana) (Z-Library).pdf"
    cc = CreateChunks(file)
    x = cc.get_clean_chunks()
    print(x[:10])