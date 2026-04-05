"""
Core data augmentation module. Takes monolingual Hindi/Bengali/Gujarati
sentences and randomly replaces 20-40% of content words with English
equivalents to generate synthetic code-mixed training data.
"""

import sys
import os
import re
import json
import random
from typing import Dict, List, Tuple, Optional
import pandas as pd

from src.exception.exception import CodeMixedSummarizationException
from src.logging.logger import logging


class SyntheticDataGenerator:
    """
    Generates synthetic code-mixed training data from monolingual
    ILSUM articles by replacing 20-40% of content words with their
    English equivalents.

    Supports:
        Hindi    + English → Hindi-English code-mixed  (<2hi>)
        Bengali  + English → Bengali-English code-mixed (<2bn>)
        Gujarati + English → Gujarati-English code-mixed (<2gu>)
        English  ILSUM     → SKIPPED (no mixing needed)

    Word selection heuristics (no POS tagger needed):
        1. Skip words with < 3 characters (grammar/function words)
        2. Skip words already in Latin script (already English)
        3. Skip words not found in dictionary (unknown content words)
        4. Replace remaining candidate words with probability
           randomly drawn from [0.20, 0.40] per sentence
        5. Skip entire sentence if < 5 words (too short to mix)

    Meaning preservation strategy:
        Dictionary-based replacement ensures semantic correctness.
        Each source word maps to ONE specific English equivalent,
        not a random English word. This preserves sentence meaning
        since "सरकार" always maps to "government", never to "tree".

    Only 'text' column gets code-mixed.
    'summary' column stays clean (model must learn to output clean English).
    """

    # Languages to process (English skipped — no mixing needed)
    SUPPORTED_LANGUAGES = ["Hindi", "Bengali", "Gujarati"]
    SKIP_LANGUAGES      = ["English"]

    # Minimum words in a sentence to attempt code-mixing
    MIN_SENTENCE_LENGTH = 5

    # Replacement probability range — drawn randomly per sentence
    MIN_REPLACE_RATIO = 0.20
    MAX_REPLACE_RATIO = 0.40

    # ------------------------------------------------------------------ #
    #  Built-in starter dictionaries (~300 most common content words)     #
    #  Format: { native_word: english_equivalent }                        #
    #  These are high-frequency nouns and adjectives from news domains    #
    #  covering politics, weather, economy, sports, science topics        #
    #  that appear most in ILSUM articles.                                #
    #                                                                     #
    #  External JSON files in artifacts/data/dictionaries/ will OVERRIDE  #
    #  these if they exist — allowing vocabulary expansion without        #
    #  changing this file.                                                #
    # ------------------------------------------------------------------ #

    _HINDI_ENGLISH: Dict[str, str] = {
        # Government & Politics
        "सरकार": "government", "राज्य": "state", "देश": "country",
        "नेता": "leader", "पार्टी": "party", "चुनाव": "election",
        "संसद": "parliament", "मंत्री": "minister", "प्रधान": "prime",
        "राष्ट्र": "nation", "नीति": "policy", "योजना": "scheme",
        "कानून": "law", "अदालत": "court", "न्याय": "justice",
        "पुलिस": "police", "सेना": "army", "सुरक्षा": "security",
        "विभाग": "department", "अधिकार": "right", "समिति": "committee",
        "अध्यक्ष": "president", "मुख्य": "chief", "केंद्र": "centre",

        # Economy & Finance
        "बाजार": "market", "व्यापार": "business", "कंपनी": "company",
        "रुपया": "rupee", "कीमत": "price", "टैक्स": "tax",
        "बैंक": "bank", "निवेश": "investment", "उत्पाद": "product",
        "आर्थिक": "economic", "विकास": "development", "उद्योग": "industry",
        "रोजगार": "employment", "वेतन": "salary", "बजट": "budget",
        "लाभ": "profit", "हानि": "loss", "ऋण": "loan",
        "शेयर": "share", "संपत्ति": "property", "आय": "income",

        # Society & Education
        "शिक्षा": "education", "विद्यालय": "school", "विश्वविद्यालय": "university",
        "छात्र": "student", "शिक्षक": "teacher", "परीक्षा": "exam",
        "स्वास्थ्य": "health", "अस्पताल": "hospital", "दवा": "medicine",
        "समाज": "society", "परिवार": "family", "महिला": "woman",
        "बच्चा": "child", "युवा": "youth", "जनसंख्या": "population",
        "धर्म": "religion", "संस्कृति": "culture", "भाषा": "language",

        # Geography & Nature
        "शहर": "city", "गांव": "village", "जिला": "district",
        "नदी": "river", "पहाड़": "mountain", "जंगल": "forest",
        "मौसम": "weather", "तापमान": "temperature", "बारिश": "rain",
        "पानी": "water", "जमीन": "land", "पर्यावरण": "environment",
        "प्रदूषण": "pollution", "ऊर्जा": "energy", "संसाधन": "resource",

        # Technology & Science
        "तकनीक": "technology", "विज्ञान": "science", "अनुसंधान": "research",
        "इंटरनेट": "internet", "कंप्यूटर": "computer", "मोबाइल": "mobile",
        "डेटा": "data", "सॉफ्टवेयर": "software", "डिजिटल": "digital",
        "अंतरिक्ष": "space", "उपग्रह": "satellite", "रॉकेट": "rocket",

        # Sports & Entertainment
        "खेल": "sport", "क्रिकेट": "cricket", "टीम": "team",
        "खिलाड़ी": "player", "मैच": "match", "जीत": "victory",
        "फिल्म": "film", "संगीत": "music", "कला": "art",

        # Common Adjectives
        "बड़ा": "big", "छोटा": "small", "नया": "new", "पुराना": "old",
        "अच्छा": "good", "बुरा": "bad", "महत्वपूर्ण": "important",
        "राष्ट्रीय": "national", "अंतरराष्ट्रीय": "international",
        "सफल": "successful", "विशेष": "special", "प्रमुख": "major",
        "स्थानीय": "local", "वैश्विक": "global", "सामाजिक": "social",
        "आधुनिक": "modern", "ऐतिहासिक": "historical", "प्राकृतिक": "natural",
    }

    _BENGALI_ENGLISH: Dict[str, str] = {
        # Government & Politics
        "সরকার": "government", "রাজ্য": "state", "দেশ": "country",
        "নেতা": "leader", "দল": "party", "নির্বাচন": "election",
        "সংসদ": "parliament", "মন্ত্রী": "minister", "জাতি": "nation",
        "নীতি": "policy", "পরিকল্পনা": "scheme", "আইন": "law",
        "আদালত": "court", "পুলিশ": "police", "সেনা": "army",
        "নিরাপত্তা": "security", "বিভাগ": "department", "কমিটি": "committee",
        "কেন্দ্র": "centre", "প্রধান": "chief", "অধিকার": "right",

        # Economy & Finance
        "বাজার": "market", "ব্যবসা": "business", "কোম্পানি": "company",
        "টাকা": "money", "মূল্য": "price", "কর": "tax",
        "ব্যাংক": "bank", "বিনিয়োগ": "investment", "পণ্য": "product",
        "অর্থনৈতিক": "economic", "উন্নয়ন": "development", "শিল্প": "industry",
        "কর্মসংস্থান": "employment", "বেতন": "salary", "বাজেট": "budget",
        "লাভ": "profit", "ক্ষতি": "loss", "ঋণ": "loan",
        "সম্পদ": "property", "আয়": "income", "শেয়ার": "share",

        # Society & Education
        "শিক্ষা": "education", "বিদ্যালয়": "school", "বিশ্ববিদ্যালয়": "university",
        "ছাত্র": "student", "শিক্ষক": "teacher", "পরীক্ষা": "exam",
        "স্বাস্থ্য": "health", "হাসপাতাল": "hospital", "ওষুধ": "medicine",
        "সমাজ": "society", "পরিবার": "family", "মহিলা": "woman",
        "শিশু": "child", "যুব": "youth", "জনসংখ্যা": "population",
        "ধর্ম": "religion", "সংস্কৃতি": "culture", "ভাষা": "language",

        # Geography & Nature
        "শহর": "city", "গ্রাম": "village", "জেলা": "district",
        "নদী": "river", "পাহাড়": "mountain", "জঙ্গল": "forest",
        "আবহাওয়া": "weather", "তাপমাত্রা": "temperature", "বৃষ্টি": "rain",
        "পানি": "water", "জমি": "land", "পরিবেশ": "environment",
        "দূষণ": "pollution", "শক্তি": "energy", "সম্পদ": "resource",

        # Technology & Science
        "প্রযুক্তি": "technology", "বিজ্ঞান": "science", "গবেষণা": "research",
        "ইন্টারনেট": "internet", "কম্পিউটার": "computer", "মোবাইল": "mobile",
        "ডেটা": "data", "সফটওয়্যার": "software", "ডিজিটাল": "digital",
        "মহাকাশ": "space", "উপগ্রহ": "satellite", "রকেট": "rocket",

        # Sports & Entertainment
        "খেলা": "sport", "ক্রিকেট": "cricket", "দল": "team",
        "খেলোয়াড়": "player", "ম্যাচ": "match", "জয়": "victory",
        "চলচ্চিত্র": "film", "সংগীত": "music", "শিল্প": "art",

        # Common Adjectives
        "বড়": "big", "ছোট": "small", "নতুন": "new", "পুরনো": "old",
        "ভালো": "good", "খারাপ": "bad", "গুরুত্বপূর্ণ": "important",
        "জাতীয়": "national", "আন্তর্জাতিক": "international",
        "সফল": "successful", "বিশেষ": "special", "প্রধান": "major",
        "স্থানীয়": "local", "বৈশ্বিক": "global", "সামাজিক": "social",
        "আধুনিক": "modern", "ঐতিহাসিক": "historical", "প্রাকৃতিক": "natural",
    }

    _GUJARATI_ENGLISH: Dict[str, str] = {
        # Government & Politics
        "સરકાર": "government", "રાજ્ય": "state", "દેશ": "country",
        "નેતા": "leader", "પક્ષ": "party", "ચૂંટણી": "election",
        "સંસદ": "parliament", "મંત્રી": "minister", "રાષ્ટ્ર": "nation",
        "નીતિ": "policy", "યોજના": "scheme", "કાયદો": "law",
        "અદાલત": "court", "પોલીસ": "police", "સેના": "army",
        "સુરક્ષા": "security", "વિભાગ": "department", "સમિતિ": "committee",
        "કેન્દ્ર": "centre", "મુખ્ય": "chief", "અધિકાર": "right",

        # Economy & Finance
        "બજાર": "market", "વ્યાપાર": "business", "કંપની": "company",
        "રૂપિયો": "rupee", "કિંમત": "price", "કર": "tax",
        "બેન્ક": "bank", "રોકાણ": "investment", "ઉત્પાદ": "product",
        "આર્થિક": "economic", "વિકાસ": "development", "ઉદ્યોગ": "industry",
        "રોજગાર": "employment", "પગાર": "salary", "બજેટ": "budget",
        "નફો": "profit", "નુકસાન": "loss", "લોન": "loan",
        "સંપત્તિ": "property", "આવક": "income", "શેર": "share",

        # Society & Education
        "શિક્ષણ": "education", "શાળા": "school", "વિશ્વવિદ્યાલય": "university",
        "વિદ્યાર્થી": "student", "શિક્ષક": "teacher", "પરીક્ષા": "exam",
        "આરોગ્ય": "health", "હોસ્પિટલ": "hospital", "દવા": "medicine",
        "સમાજ": "society", "પરિવાર": "family", "મહિલા": "woman",
        "બાળક": "child", "યુવા": "youth", "વસ્તી": "population",
        "ધર્મ": "religion", "સંસ્કૃતિ": "culture", "ભાષા": "language",

        # Geography & Nature
        "શહેર": "city", "ગામ": "village", "જિલ્લો": "district",
        "નદી": "river", "પહાડ": "mountain", "જંગલ": "forest",
        "હવામાન": "weather", "તાપમાન": "temperature", "વરસાદ": "rain",
        "પાણી": "water", "જમીન": "land", "પર્યાવરણ": "environment",
        "પ્રદૂષણ": "pollution", "ઊર્જા": "energy", "સંસાધન": "resource",

        # Technology & Science
        "ટેકનોલોજી": "technology", "વિજ્ઞાન": "science", "સંશોધન": "research",
        "ઈન્ટરનેટ": "internet", "કોમ્પ્યુટર": "computer", "મોબાઈલ": "mobile",
        "ડેટા": "data", "સોફ્ટવેર": "software", "ડિજિટલ": "digital",
        "અવકાશ": "space", "ઉપગ્રહ": "satellite", "રોકેટ": "rocket",

        # Sports & Entertainment
        "રમત": "sport", "ક્રિકેટ": "cricket", "ટીમ": "team",
        "ખેલાડી": "player", "મેચ": "match", "જીત": "victory",
        "ફિલ્મ": "film", "સંગીત": "music", "કલા": "art",

        # Common Adjectives
        "મોટો": "big", "નાનો": "small", "નવો": "new", "જૂનો": "old",
        "સારો": "good", "ખરાબ": "bad", "મહત્વપૂર્ણ": "important",
        "રાષ્ટ્રીય": "national", "આંતરરાષ્ટ્રીય": "international",
        "સફળ": "successful", "વિશેષ": "special", "મુખ્ય": "major",
        "સ્થાનિક": "local", "વૈશ્વિક": "global", "સામાજિક": "social",
        "આધુનિક": "modern", "ઐતિહાસિક": "historical", "કુદરતી": "natural",
    }

    # Map language name → built-in dictionary
    _BUILTIN_DICTS = {
        "Hindi":    "_HINDI_ENGLISH",
        "Bengali":  "_BENGALI_ENGLISH",
        "Gujarati": "_GUJARATI_ENGLISH",
    }

    def __init__(
        self,
        dict_dir: str  = "artifacts/data/dictionaries",
        output_dir: str = "artifacts/data/synthetic_codemixed",
        min_sentence_length: int   = 5,
        min_replace_ratio: float   = 0.20,
        max_replace_ratio: float   = 0.40,
        random_seed: int           = 42,
    ):
        """
        Initialize SyntheticDataGenerator.

        Args:
            dict_dir            : Directory containing external JSON dictionaries.
                                  If JSON files exist here they OVERRIDE built-in dicts.
                                  Expected files:
                                    hindi_english.json
                                    bengali_english.json
                                    gujarati_english.json
            output_dir          : Where to save synthetic CSVs.
            min_sentence_length : Skip sentences with fewer words than this.
            min_replace_ratio   : Lower bound of replacement probability range.
            max_replace_ratio   : Upper bound of replacement probability range.
            random_seed         : Seed for reproducibility.
        """
        try:
            self.dict_dir             = dict_dir
            self.output_dir           = output_dir
            self.min_sentence_length  = min_sentence_length
            self.min_replace_ratio    = min_replace_ratio
            self.max_replace_ratio    = max_replace_ratio

            random.seed(random_seed)
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.dict_dir,   exist_ok=True)

            # Load dictionaries — external JSON overrides built-in if exists
            self.dictionaries = self._load_dictionaries()

            logging.info(
                f"SyntheticDataGenerator initialized\n"
                f"  dict_dir     : {dict_dir}\n"
                f"  output_dir   : {output_dir}\n"
                f"  replace_ratio: {min_replace_ratio}–{max_replace_ratio}\n"
                f"  min_sent_len : {min_sentence_length}\n"
                f"  Dictionary sizes loaded:\n" +
                "\n".join(
                    f"    {lang}: {len(d)} words"
                    for lang, d in self.dictionaries.items()
                )
            )
        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Dictionary loading                                                  #
    # ------------------------------------------------------------------ #

    def _load_dictionaries(self) -> Dict[str, Dict[str, str]]:
        """
        Load word translation dictionaries for all supported languages.
        Priority: external JSON file > built-in class dictionary.

        External JSON files must be at:
            artifacts/data/dictionaries/hindi_english.json
            artifacts/data/dictionaries/bengali_english.json
            artifacts/data/dictionaries/gujarati_english.json

        Returns:
            Dict mapping language name → { native_word: english_word }
        """
        try:
            json_filenames = {
                "Hindi":    "hindi_english.json",
                "Bengali":  "bengali_english.json",
                "Gujarati": "gujarati_english.json",
            }
            loaded = {}

            for lang in self.SUPPORTED_LANGUAGES:
                json_path = os.path.join(self.dict_dir, json_filenames[lang])

                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        loaded[lang] = json.load(f)
                    logging.info(
                        f"  [{lang}] Loaded external dictionary: "
                        f"{json_path} ({len(loaded[lang])} words)"
                    )
                else:
                    # Fall back to built-in dictionary
                    builtin_attr = self._BUILTIN_DICTS[lang]
                    loaded[lang] = getattr(self, builtin_attr).copy()
                    logging.info(
                        f"  [{lang}] Using built-in dictionary "
                        f"({len(loaded[lang])} words). "
                        f"Place {json_filenames[lang]} in {self.dict_dir}/ "
                        f"to override with a larger vocabulary."
                    )

            return loaded

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    def save_dictionaries_to_json(self) -> Dict[str, str]:
        """
        Save the currently loaded dictionaries to JSON files in dict_dir.
        Useful for inspecting or extending the built-in word lists.

        Returns:
            Dict mapping language → saved JSON file path
        """
        try:
            paths = {}
            json_filenames = {
                "Hindi":    "hindi_english.json",
                "Bengali":  "bengali_english.json",
                "Gujarati": "gujarati_english.json",
            }
            for lang, dictionary in self.dictionaries.items():
                path = os.path.join(self.dict_dir, json_filenames[lang])
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(dictionary, f, ensure_ascii=False, indent=2)
                paths[lang] = path
                logging.info(f"  Saved {lang} dictionary → {path}")
            return paths

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Word-level helpers                                                  #
    # ------------------------------------------------------------------ #

    def _is_latin_script(self, word: str) -> bool:
        """
        Check if a word is already in Latin/English script.
        These are skipped — no point replacing an already-English word.
        """
        return bool(re.match(r'^[a-zA-Z]+$', word))

    def _is_too_short(self, word: str) -> bool:
        """
        Check if a word is too short to be a content word.
        Words < 3 characters are almost always function/grammar words:
            Hindi   : है, की, का, में, से, पर, और, को
            Bengali : এর, এ, এই, ও, তে
            Gujarati: છે, ની, નો, માં, થી
        """
        return len(word) < 3

    def _is_punctuation(self, word: str) -> bool:
        """Check if token is purely punctuation — skip these."""
        return bool(re.match(r'^[^\w]+$', word))

    def _is_candidate(self, word: str, language: str) -> bool:
        """
        Determine if a word is a valid candidate for English replacement.
        A word is a candidate ONLY if ALL conditions are met:
            1. Not too short (< 3 chars)
            2. Not already in Latin script
            3. Not punctuation
            4. Exists in the language dictionary
               (dictionary lookup ensures semantic correctness —
                only words with known English equivalents are replaced,
                preventing meaning loss from random substitution)

        Args:
            word    : Single word token from the sentence
            language: Language name e.g. 'Hindi'

        Returns:
            True if word can be replaced with its English equivalent
        """
        if self._is_too_short(word):
            return False
        if self._is_latin_script(word):
            return False
        if self._is_punctuation(word):
            return False
        if word not in self.dictionaries.get(language, {}):
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Sentence-level mixing                                               #
    # ------------------------------------------------------------------ #

    def _mix_sentence(self, text: str, language: str) -> Tuple[str, int, int]:
        """
        Apply code-mixing to a single sentence.

        Strategy:
            1. Tokenize by whitespace
            2. Find all candidate words (via _is_candidate)
            3. Draw a replacement probability p from [min, max] range
            4. For each candidate, replace with English with probability p
            5. Reconstruct sentence preserving original spacing

        This means:
            - Grammar words are ALWAYS preserved (too short or not in dict)
            - Only semantically known content words get replaced
            - Probability is per-sentence not per-word (natural variation)
            - Sentence structure is completely unchanged

        Args:
            text    : Input sentence string
            language: Language name

        Returns:
            Tuple of (mixed_text, num_candidates, num_replaced)
        """
        try:
            words = text.split()

            if len(words) < self.min_sentence_length:
                return text, 0, 0

            # Find candidate positions
            candidate_indices = [
                i for i, w in enumerate(words)
                if self._is_candidate(w, language)
            ]

            if not candidate_indices:
                return text, 0, 0

            # Draw sentence-level replacement probability
            replace_prob = random.uniform(
                self.min_replace_ratio,
                self.max_replace_ratio
            )

            # Replace candidates probabilistically
            num_replaced = 0
            for idx in candidate_indices:
                if random.random() < replace_prob:
                    original_word          = words[idx]
                    words[idx]             = self.dictionaries[language][original_word]
                    num_replaced          += 1

            mixed_text = " ".join(words)
            return mixed_text, len(candidate_indices), num_replaced

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  DataFrame-level generation                                          #
    # ------------------------------------------------------------------ #

    def generate_from_dataframe(
        self,
        df: pd.DataFrame,
        language: str,
    ) -> pd.DataFrame:
        """
        Generate synthetic code-mixed data from a preprocessed DataFrame.

        Input DataFrame columns (from preprocessor.py output):
            text    : Cleaned article text WITH language tag prepended
                      e.g. "<2hi> सरकार ने नई योजना शुरू की"
            summary : Clean summary (NOT tagged, NOT code-mixed)
            language: Language name e.g. "Hindi"

        Processing:
            1. Strip language tag from text before mixing
               (tag is re-added after mixing to avoid replacing the tag itself)
            2. Apply _mix_sentence to text column only
            3. Re-prepend language tag to mixed text
            4. Summary column is left completely unchanged

        Output DataFrame columns:
            text    : Code-mixed text WITH language tag
                      e.g. "<2hi> government ने नई scheme शुरू की"
            summary : Unchanged clean summary
            language: Unchanged language name

        Args:
            df      : Preprocessed DataFrame from preprocessor.py
            language: Language name — must be in SUPPORTED_LANGUAGES

        Returns:
            DataFrame with code-mixed text column
        """
        try:
            if language in self.SKIP_LANGUAGES:
                logging.info(f"  [{language}] Skipping — English needs no mixing")
                return pd.DataFrame()

            if language not in self.SUPPORTED_LANGUAGES:
                raise ValueError(
                    f"Unsupported language '{language}'. "
                    f"Supported: {self.SUPPORTED_LANGUAGES}"
                )

            logging.info(
                f"  [{language}] Generating synthetic data "
                f"from {len(df)} rows..."
            )

            df = df.copy()

            # Extract language tag from text column
            # Format is "<2hi> actual text..." — split on first space
            tag_pattern = re.compile(r'^(<2\w+>)\s+')

            mixed_texts     = []
            skipped         = 0
            total_candidates = 0
            total_replaced   = 0

            for _, row in df.iterrows():
                text = str(row["text"])

                # Separate tag from text content
                tag_match = tag_pattern.match(text)
                if tag_match:
                    tag          = tag_match.group(1)
                    text_content = text[tag_match.end():]
                else:
                    tag          = ""
                    text_content = text

                # Apply code-mixing to text content only
                mixed_content, n_candidates, n_replaced = self._mix_sentence(
                    text_content, language
                )

                # Track stats
                if n_candidates == 0:
                    skipped += 1

                total_candidates += n_candidates
                total_replaced   += n_replaced

                # Re-attach tag
                mixed_text = f"{tag} {mixed_content}".strip() if tag else mixed_content
                mixed_texts.append(mixed_text)

            df["text"] = mixed_texts

            # Drop rows where no mixing happened at all
            # (sentence was too short or had no dictionary matches)
            original_len = len(df)
            df = df[df["text"].str.contains(
                r'[a-zA-Z].*[\u0900-\u097F\u0980-\u09FF\u0A80-\u0AFF]|'
                r'[\u0900-\u097F\u0980-\u09FF\u0A80-\u0AFF].*[a-zA-Z]',
                regex=True
            )].reset_index(drop=True)

            actual_mixed = len(df)
            logging.info(
                f"  [{language}] Generation complete:\n"
                f"    Input rows       : {original_len}\n"
                f"    Successfully mixed: {actual_mixed}\n"
                f"    Skipped (no match): {skipped}\n"
                f"    Total candidates  : {total_candidates}\n"
                f"    Total replaced    : {total_replaced}\n"
                f"    Avg replace rate  : "
                f"{(total_replaced/max(total_candidates,1)*100):.1f}%"
            )

            return df

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Save                                                                #
    # ------------------------------------------------------------------ #

    def _save_synthetic_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        language: str,
    ) -> Dict[str, str]:
        """
        Save synthetic train/val/test splits for one language.

        Output paths:
            artifacts/data/synthetic_codemixed/Hindi/train_synthetic.csv
            artifacts/data/synthetic_codemixed/Hindi/val_synthetic.csv
            artifacts/data/synthetic_codemixed/Hindi/test_synthetic.csv
        """
        try:
            lang_dir = os.path.join(self.output_dir, language)
            os.makedirs(lang_dir, exist_ok=True)

            paths = {}
            for split_name, df in [
                ("train", train_df),
                ("val",   val_df),
                ("test",  test_df),
            ]:
                path = os.path.join(lang_dir, f"{split_name}_synthetic.csv")
                df.to_csv(path, index=False)
                paths[split_name] = path
                logging.info(f"    Saved {split_name}: {path} ({len(df)} rows)")

            return paths

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)

    # ------------------------------------------------------------------ #
    #  Main pipeline                                                       #
    # ------------------------------------------------------------------ #

    def initiate_synthetic_generation(
        self,
        preprocessed_artifacts: Dict,
    ) -> Dict[str, Dict[str, str]]:
        """
        Main pipeline entry point. Generates synthetic code-mixed data
        from all preprocessed ILSUM language splits.

        Takes the output of TextPreprocessor.initiate_preprocessing() directly.

        Args:
            preprocessed_artifacts: Dict from preprocessor output:
                {
                    'ilsum': {
                        'Hindi':    { 'train': path, 'val': path, 'test': path },
                        'Bengali':  { ... },
                        'English':  { ... },   ← skipped automatically
                        'Gujarati': { ... },
                    },
                    'hinge': { ... }             ← skipped (already real code-mixed)
                }

        Returns:
            Dict mapping language → split paths:
                {
                    'Hindi':    { 'train': path, 'val': path, 'test': path },
                    'Bengali':  { 'train': path, 'val': path, 'test': path },
                    'Gujarati': { 'train': path, 'val': path, 'test': path },
                }

        Output folder structure:
            artifacts/data/synthetic_codemixed/
            ├── Hindi/
            │   ├── train_synthetic.csv
            │   ├── val_synthetic.csv
            │   └── test_synthetic.csv
            ├── Bengali/
            │   ├── train_synthetic.csv
            │   ├── val_synthetic.csv
            │   └── test_synthetic.csv
            └── Gujarati/
                ├── train_synthetic.csv
                ├── val_synthetic.csv
                └── test_synthetic.csv
        """
        try:
            logging.info("=" * 50)
            logging.info("Starting synthetic data generation pipeline...")
            logging.info("=" * 50)

            artifact = {}

            for lang, split_paths in preprocessed_artifacts["ilsum"].items():

                # Skip English — no meaningful mixing possible
                if lang in self.SKIP_LANGUAGES:
                    logging.info(f"Skipping [{lang}] — English needs no mixing")
                    continue

                logging.info(f"Processing [{lang}]...")

                # Load each split, generate synthetic version
                split_dfs = {}
                for split_name, csv_path in split_paths.items():
                    logging.info(
                        f"  [{lang}/{split_name}] "
                        f"reading: {csv_path}"
                    )
                    df = pd.read_csv(csv_path)
                    split_dfs[split_name] = self.generate_from_dataframe(
                        df, language=lang
                    )

                # Save all splits for this language
                artifact[lang] = self._save_synthetic_splits(
                    train_df=split_dfs["train"],
                    val_df=split_dfs["val"],
                    test_df=split_dfs["test"],
                    language=lang,
                )

                logging.info(
                    f"[{lang}] synthetic generation complete — "
                    f"saved to {os.path.join(self.output_dir, lang)}"
                )

            logging.info("=" * 50)
            logging.info("Synthetic generation pipeline completed!")
            logging.info("Synthetic artifact summary:")
            for lang, paths in artifact.items():
                for split, path in paths.items():
                    logging.info(f"  {lang}/{split}: {path}")
            logging.info("=" * 50)

            return artifact

        except Exception as e:
            raise CodeMixedSummarizationException(e, sys)


if __name__ == "__main__":
    try:
        from src.components.data.dataset_loader import DatasetLoader
        from src.components.data.preprocessor import TextPreprocessor

        print("Step 1: Loading datasets...")
        loader         = DatasetLoader()
        data_artifacts = loader.initiate_data_loading()

        print("\nStep 2: Preprocessing...")
        preprocessor        = TextPreprocessor()
        clean_artifacts     = preprocessor.initiate_preprocessing(data_artifacts)

        print("\nStep 3: Generating synthetic code-mixed data...")
        generator           = SyntheticDataGenerator()
        synthetic_artifacts = generator.initiate_synthetic_generation(clean_artifacts)

        print("\n" + "=" * 50)
        print("Synthetic Data Artifacts:")
        print("=" * 50)
        for lang, paths in synthetic_artifacts.items():
            print(f"\n{lang}:")
            for split, path in paths.items():
                print(f"  {split}: {path}")

        print("\nSynthetic generation completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()