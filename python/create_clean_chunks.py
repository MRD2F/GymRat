from pathlib import Path
import sys
sys.path.append(str(Path('../python').resolve()))
from clean_text import CleanText
import unicodedata

# import unicodedata
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Docling chunker
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

#Read a .pdf file, and splits it into chunks. The chunks are processed to clean them to create an
#optimal input for content generation using a LLM model
#The chucks can be saved, or loaded directly into the GenerateQAContent class from the create_qa.py script


#######################################################################################################
########### langchain.text_splitter import RecursiveCharacterTextSplitter #############################
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
    input_file="../data/books/rebuilding_milo.pdf"
    output_chunks_file = "../data/rebuilding_milo_chunks_docling_max_tokens128_min_tokens50_meta_llama3p18B.txt"
    model_id = "meta-llama/Llama-3.1-8B-Instruct" #TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    def llama_token_len(text):
        return len(tokenizer.encode(text))

    save_docling_chunks=False
    cc_langchain = CreateChunks(file_path=input_file, chunk_size=500, chunk_overlap=50, length_function=llama_token_len)
    y = cc_langchain.get_clean_chunks()
    print(y[:10])





#######################################################################################################
########################## Docling and costum Hibrid Splitter #########################################
#######################################################################################################

class HeaderPreservingChunker:
    """
    Wraps HybridChunker but ensures short lines (headers, captions, etc.)
    are merged into the following content chunk.
    """

    def __init__(self, file_path, tokenizer, max_tokens=512, min_tokens=50, short_threshold=40):
        self.file_path = file_path
        self.base_chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            respect_sentence_boundaries=True,
            merge_peers=True,
            merge_tiny=True
        )
        self.short_threshold = short_threshold

        self.chunks = self.create_chunks(self.load_document().document)
        #self.chunks = self.create_chunks(results)
        self.non_ascii_words, self.private_unicode_words, self.non_ascii_chars  = self.get_problematic_lists(self.chunks)

    def load_document(self):
        """
        Load the document from the specified file path.
        """
        converter = DocumentConverter()
        return converter.convert(self.file_path)

    def create_chunks(self, dl_doc):
        """
        Chunk the doc and merge orphan short lines into the next chunk.
        """
        raw_chunks = list(self.base_chunker.chunk(dl_doc=dl_doc))

        merged_chunks = []
        buffer = None

        for chunk in raw_chunks:
            text = chunk.text.strip()

            # If chunk is very short (potential header), buffer it
            if len(text) <= self.short_threshold and "\n" not in text:
                if buffer is None:
                    buffer = text
                else:
                    buffer += " " + text
                continue

            # If we have a buffered header, prepend it to this chunk
            if buffer:
                new_text = buffer + "\n" + text
                chunk.text = new_text
                buffer = None

            merged_chunks.append(chunk)

        # If only orphan headers remain at the end, merge them into the last chunk
        if buffer and merged_chunks:
            merged_chunks[-1].text = buffer + "\n" + merged_chunks[-1].text

        return merged_chunks

    def get_problematic_lists(self, chunks):
        # Create lists with problematic words or characters for all the chunks in the provided file
        # Used later for the CleanText class to remove or fix this words. Also used manually for debbuging and checking the quality of the resulting cleaned chunks 
        non_ascii_words, private_unicode_words, non_ascii_chars = set(), set(), set()
        for chunk in chunks:
            #Normalize text for consistent Unicode handling
            text = unicodedata.normalize("NFKC", chunk.text)

            clean_text = CleanText(text)

            non_ascii_words.update(clean_text.find_words_with_non_ascii())
            private_unicode_words.update(clean_text.find_words_with_private_unicode())
            non_ascii_chars.update(clean_text.extract_non_ascii_characters())

        non_ascii_words = list(sorted(non_ascii_words))
        private_unicode_words = list(sorted(private_unicode_words))
        non_ascii_chars = list(sorted(non_ascii_chars))

        return non_ascii_words, private_unicode_words, non_ascii_chars

    def get_clean_text(self, text):
        return CleanText(text, self.non_ascii_words)

    def clean_chunk(self, text):
        clean_text = self.get_clean_text(text)
        words = text.split() # chunk == text
        clean_chunk = ""
        for word in words:
            #new_text =  self.clean_word(word)
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
            text_chunk = chunk.text 
            print(f"Processing chuck number: {n}")
            text_chunk = self.clean_chunk(text_chunk)
            text_chunks.append(text_chunk)
            n+=1
        return text_chunks

text_docling_chunker = False
save_docling_chunks = False
input_file="../data/books/rebuilding_milo.pdf"
output_chunks_file = "../data/rebuilding_milo_chunks_docling_max_tokens128_min_tokens50_meta_llama3p18B.txt"
model_id = "meta-llama/Llama-3.1-8B-Instruct" #TinyLlama/TinyLlama-1.1B-Chat-v1.0"

if text_docling_chunker:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    cc = HeaderPreservingChunker(file_path=input_file, tokenizer=tokenizer,  max_tokens=128)
    x = cc.get_clean_chunks()
    print(x[:10])

    if save_docling_chunks:
        with open(output_chunks_file, "w", encoding="utf-8") as f:
            for item in x:
                f.write(item + "\n")
