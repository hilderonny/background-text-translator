PROGRAM_VERSION = "1.0.0"

import time
import os
import json
import datetime
import shutil
import stat
import sys
import argparse
import glob
import re
import PyPDF2
from typing import Sequence

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--inputpath", "-i", type=str, action="store", required=True, help="Directory where the text files to process are obtained from. Must be writable.")
parser.add_argument("--processingpath", "-p", type=str, action="store", required=True, help="Directory where the currently processed text file gets stored. Must be writable.")
parser.add_argument("--outputpath", "-o", type=str, action="store", required=True, help="Directory where the output JSON files will be stored. Must be writable.")
parser.add_argument("--stanzapath", "-s", type=str, action="store", required=True, help="Directory where the Stanza models are stored.")
parser.add_argument("--huggingfacepath", "-f", type=str, action="store", required=True, help="Directory where the HuggingFace models are stored.")
parser.add_argument("--usegpu", "-g", action="store_true", help="Use GPU for neural network calculations.")
parser.add_argument("--version", "-v", action="version", version=PROGRAM_VERSION)
args = parser.parse_args()

# Map program parameters
INPUTPATH = args.inputpath
PROCESSINGPATH = args.processingpath
OUTPUTPATH = args.outputpath
STANZAPATH = args.stanzapath
HUGGINGFACEPATH = args.huggingfacepath
USEGPU = args.usegpu

# Check existence of directories
if not os.access(INPUTPATH, os.R_OK | os.W_OK):
    sys.exit(f"ERROR: Cannot read and write input directory {INPUTPATH}")
if not os.access(PROCESSINGPATH, os.R_OK | os.W_OK):
    sys.exit(f"ERROR: Cannot read and write processing directory {PROCESSINGPATH}")
if not os.access(OUTPUTPATH, os.R_OK | os.W_OK):
    sys.exit(f"ERROR: Cannot read and write output directory {OUTPUTPATH}")
if not os.access(STANZAPATH, os.R_OK):
    sys.exit(f"ERROR: Cannot read Stanza directory {STANZAPATH}")
if not os.access(HUGGINGFACEPATH, os.R_OK):
    sys.exit(f"ERROR: Cannot read HuggingFace directory {HUGGINGFACEPATH}")

# Define environment variables
os.environ["HF_HOME"] = HUGGINGFACEPATH
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["STANZA_RESOURCES_DIR"] = STANZAPATH

# Load Stanza
print("Loading Stanza")
import stanza

# Load HuggingFace
print("Loading HuggingFace")
from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self, source_lang: str, dest_lang: str, use_gpu: bool=False) -> None:
        self.use_gpu = use_gpu
        self.model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}"
        self.model = MarianMTModel.from_pretrained(self.model_name)
        if use_gpu:
            self.model = self.model.cuda()
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
    def translate(self, texts: Sequence[str]) -> Sequence[str]:
        tokens = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        if self.use_gpu:
            tokens = {k:v.cuda() for k, v in tokens.items()}
        translate_tokens = self.model.generate(**tokens)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translate_tokens]

def process_text(source_text, source_language, target_language, indirect_translation):
    # Split text into smaller parts
    nlp = stanza.Pipeline(source_language, processors="tokenize", use_gpu=USEGPU)
    nlp_result = nlp.process(source_text)
    detected_sentences = list(map(lambda sentence: sentence.text, nlp_result.sentences))
    sentences_to_process = []
    for sentence in detected_sentences:
        if len(sentence) < 600:
            if len(sentence) > 1:
                sentences_to_process.append(sentence)
        else:
            # Sätze erscheinen zu lang. Aufteilung nach Punktuation
            for sentence_part in re.split("\.|\?|\!|\:", sentence):
                if len(sentence_part) > 600:
                    # Immernoch zu lang, Aufteilung nach Zeilenumbrüchen
                    for line in sentence_part.splitlines():
                        if len(line) > 1:
                            # Prinzipiell nur die ersten 600 Zeichen nehmen
                            sentences_to_process.append(line[:600])
                elif len(sentence_part) > 1:
                    sentences_to_process.append(sentence_part)
    # Translate text parts
    if indirect_translation == True:
        translator_source_to_english = Translator(source_language, "en", USEGPU)
        translator_english_to_target = Translator("en", target_language, USEGPU)
        english_sentences = translator_source_to_english.translate(sentences_to_process)
        return "\n".join(translator_english_to_target.translate(english_sentences))
    else:
        translator = Translator(source_language, target_language, USEGPU)
        return "\n".join(translator.translate(sentences_to_process))

def process_text_file(file_path, source_language, target_language, indirect_translation, result):
    with open(file_path, "r") as text_file:
        result["pages"] = []
        source_text = text_file.read()
        translated_text = process_text(source_text, source_language, target_language, indirect_translation)
        result["pages"].append(translated_text)

def process_pdf_file(file_path, source_language, target_language, indirect_translation, result):
    with open(file_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        # In Seiten separieren
        num_pages = len(pdf_reader.pages)
        result["pages"] = []
        for i in range(num_pages):
            page = pdf_reader.pages[i]
            source_text = page.extract_text()
            translated_text = process_text(source_text, source_language, target_language, indirect_translation)
            result["pages"].append(translated_text)

def process_file(file_path, file_type, source_language, target_language):
    start_time = datetime.datetime.now()
    result = {}
    try:
        # Check directories for translation models
        stanza_source_language_path = os.path.join(STANZAPATH, source_language)
        if not os.access(stanza_source_language_path, os.R_OK):
            raise Exception(f"ERROR: Cannot read Stanza language directory {stanza_source_language_path}")
        huggingface_direct_translation_path = os.path.join(HUGGINGFACEPATH, "hub", f"models--Helsinki-NLP--opus-mt-{source_language}-{target_language}")
        huggingface_from_en_translation_path = os.path.join(HUGGINGFACEPATH, "hub", f"models--Helsinki-NLP--opus-mt-en-{target_language}")
        huggingface_to_en_translation_path= os.path.join(HUGGINGFACEPATH, "hub", f"models--Helsinki-NLP--opus-mt-{source_language}-en")
        indirect_translation = False
        if not os.access(huggingface_direct_translation_path, os.R_OK):
            if not os.access(huggingface_from_en_translation_path, os.R_OK) or not os.access(huggingface_to_en_translation_path, os.R_OK):
                raise Exception(f"ERROR: Cannot read Huggingface language directories {huggingface_from_en_translation_path} or ({huggingface_to_en_translation_path} and {huggingface_from_en_translation_path})")
            indirect_translation = True
        print("Processing file " + file_path)
        if file_type == "pdf":
            process_pdf_file(file_path, source_language, target_language, indirect_translation, result)
        elif file_type == "txt":
            process_text_file(file_path, source_language, target_language, indirect_translation, result)
        else:
            raise Exception(f"ERROR: File type '{file_type}' is currently not supported.")
    except Exception as ex:
        print(ex)
        result["exception"] = str(ex)
    finally:
        print("Deleting file " + file_path)
        os.remove(file_path)
        pass
    end_time = datetime.datetime.now()
    result["duration"] = (end_time - start_time).total_seconds()
    return result

def check_and_process_files():
    file_was_processed = False
    for input_json_file_path in glob.glob(os.path.join(INPUTPATH, "*.jsonmetadata")):
        print("Processing " + input_json_file_path)
        input_json_file_name = os.path.basename(input_json_file_path)
        print("File Name " + input_json_file_name)
        input_json_file_path = os.path.join(INPUTPATH, input_json_file_name)
        if os.path.isfile(input_json_file_path):
            try:
                # Erst mal Datei aus INPUT Verzeichnis bewegen, damit andere Prozesse diese nicht ebenfalls verarbeiten
                processing_json_file_path = os.path.join(PROCESSINGPATH, input_json_file_name)
                shutil.move(input_json_file_path, processing_json_file_path)
                os.chmod(processing_json_file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO ) # Let the background process delete the file afterwards
                # Metadaten aus JSON extrahieren
                with open(processing_json_file_path, "r", encoding="utf-8") as pf:
                    metadata = json.load(pf)
                source_language = metadata["source"]
                target_language = metadata["target"]
                file_name = metadata["file"]
                file_type = metadata["type"]
                # Datei verarbeiten
                input_file_path = os.path.join(INPUTPATH, file_name);
                processing_file_path = os.path.join(PROCESSINGPATH, file_name);
                shutil.move(input_file_path, processing_file_path)
                os.chmod(processing_file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO )
                result = process_file(processing_file_path, file_type, source_language, target_language)
                json_result = json.dumps(result, indent=2)
                output_file_path = os.path.join(OUTPUTPATH, file_name + ".json")
                print("Writing output file " + output_file_path)
                output_file = os.open(output_file_path, os.O_RDWR|os.O_CREAT)
                os.write(output_file, str.encode(json_result))
                os.close(output_file)
                print(json_result)
                file_was_processed = True
                return file_was_processed # Let the program wait a moment and recheck the uplopad directory
            except Exception as ex:
                print(ex)
            finally: # Hat nicht geklappt. Eventuell hat ein anderer Prozess die Datei bereits weg geschnappt. Egal.
                return
    return file_was_processed

try:
    print("Ready and waiting for action")
    while True:
        file_was_processed = check_and_process_files()
        if file_was_processed == False:
            time.sleep(3)
finally:
    pass
