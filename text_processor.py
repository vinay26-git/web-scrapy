"""
Advanced text preprocessing module for web scraped content.
Handles cleaning, normalization, and preparation for chunking.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Tuple
from html import unescape
import string

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

from loguru import logger


class TextProcessor:
    """Advanced text processor with content-aware cleaning."""
    
    def __init__(self):
        self._setup_nltk()
        self._setup_spacy()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def _setup_nltk(self):
        """Download required NLTK data."""
        required_data = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except:
                    logger.warning(f"Could not download NLTK data: {data}")
    
    def _setup_spacy(self):
        """Setup spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy en_core_web_sm model not found. Some features may be limited.")
            self.nlp = None
    
    def clean_html_artifacts(self, text: str) -> str:
        """Remove HTML artifacts and decode entities."""
        # Decode HTML entities
        text = unescape(text)
        
        # Remove HTML tags (in case any remain)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Remove script and style content
        text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace common Unicode characters
        replacements = {
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201C': '"',  # Left double quotation mark
            '\u201D': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...', # Horizontal ellipsis
            '\u00A0': ' ',  # Non-breaking space
            '\u00AD': '',   # Soft hyphen
        }
        
        for unicode_char, replacement in replacements.items():
            text = text.replace(unicode_char, replacement)
        
        return text
    
    def clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        
        # Remove empty lines and join
        lines = [line for line in lines if line]
        
        # Join lines with proper spacing
        text = '\n'.join(lines)
        
        # Clean up multiple newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def remove_navigation_artifacts(self, text: str) -> str:
        """Remove common navigation and UI artifacts."""
        patterns_to_remove = [
            # Navigation patterns
            r'\b(Home|About|Contact|Privacy|Terms)\s*\|',
            r'\b(Home|About|Contact)\s*>',
            r'Skip to content',
            r'Skip to main content',
            r'Skip navigation',
            
            # Social media patterns
            r'Share on (Facebook|Twitter|LinkedIn|Instagram)',
            r'Follow us on',
            r'Like us on',
            r'Subscribe to',
            
            # Common UI elements
            r'Read more',
            r'Continue reading',
            r'Click here',
            r'Learn more',
            r'View all',
            r'Show more',
            
            # Cookie/GDPR notices
            r'This website uses cookies',
            r'We use cookies',
            r'Accept cookies',
            r'Cookie policy',
            
            # Advertisement patterns
            r'Advertisement',
            r'Sponsored content',
            r'Ads by',
            
            # Pagination
            r'Page \d+ of \d+',
            r'Previous\s+Next',
            r'\d+\s+of\s+\d+',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def remove_boilerplate_content(self, text: str, site_type: str) -> str:
        """Remove site-specific boilerplate content."""
        if site_type == "news":
            # News-specific patterns
            patterns = [
                r'Breaking news:?\s*',
                r'Last updated:?\s*\d+.*',
                r'Published:?\s*\d+.*',
                r'Reporter:?\s*\w+.*',
                r'Contact the reporter',
                r'Send tips to',
                r'More stories like this',
                r'Related articles?:?',
                r'Read the full story',
                r'Subscribe to our newsletter',
                r'Sign up for alerts',
            ]
        elif site_type == "wikipedia":
            # Wikipedia-specific patterns
            patterns = [
                r'\[edit\]',
                r'\[citation needed\]',
                r'\[clarification needed\]',
                r'\[\d+\]',  # Reference numbers
                r'Coordinates:?\s*\d+',
                r'From Wikipedia, the free encyclopedia',
                r'Jump to navigation',
                r'Jump to search',
                r'This article needs additional citations',
                r'This article may require cleanup',
            ]
        else:  # business
            # Business site patterns
            patterns = [
                r'Contact us',
                r'Get in touch',
                r'Free consultation',
                r'Request a quote',
                r'Learn more about our',
                r'Our services include',
                r'Call us today',
                r'Schedule a meeting',
                r'Download our',
                r'Sign up for our newsletter',
            ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract and clean sentences from text."""
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback if NLTK fails
            sentences = re.split(r'[.!?]+', text)
        
        # Clean sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip very short sentences
            if len(sentence) < 10:
                continue
            
            # Skip sentences that are mostly numbers or special characters
            alpha_chars = sum(c.isalpha() for c in sentence)
            if alpha_chars < len(sentence) * 0.6:
                continue
            
            # Skip sentences with too many uppercase words (likely headers)
            words = sentence.split()
            upper_words = sum(1 for word in words if word.isupper() and len(word) > 1)
            if len(words) > 0 and upper_words / len(words) > 0.5:
                continue
            
            cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases using NLP."""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text[:1000000])  # Limit text length for spaCy
            
            key_phrases = []
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip().lower()
                if len(phrase) > 3 and len(phrase.split()) <= 4:
                    key_phrases.append(phrase)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                    phrase = ent.text.strip().lower()
                    if len(phrase) > 2:
                        key_phrases.append(phrase)
            
            # Remove duplicates and return
            return list(set(key_phrases))
        except:
            logger.warning("Failed to extract key phrases with spaCy")
            return []
    
    def calculate_text_statistics(self, text: str) -> Dict[str, any]:
        """Calculate various text statistics."""
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        # Basic statistics
        stats = {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(text.split('\n\n')),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'avg_chars_per_word': len(text) / len(words) if words else 0,
        }
        
        # Readability metrics (simplified)
        if sentences and words:
            # Flesch Reading Ease (simplified)
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
            stats['flesch_reading_ease'] = max(0, min(100, flesch_score))
        else:
            stats['flesch_reading_ease'] = 0
        
        return stats
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = True
            else:
                previous_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def process_content(self, content: str, site_type: str, metadata: Dict = None) -> Dict[str, any]:
        """
        Main processing pipeline for scraped content.
        
        Args:
            content: Raw scraped content
            site_type: Type of site (business, news, wikipedia)
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary containing processed content and analysis
        """
        logger.info(f"Processing {len(content)} characters of {site_type} content")
        
        # Step 1: Clean HTML artifacts
        processed_text = self.clean_html_artifacts(content)
        
        # Step 2: Normalize Unicode
        processed_text = self.normalize_unicode(processed_text)
        
        # Step 3: Remove navigation artifacts
        processed_text = self.remove_navigation_artifacts(processed_text)
        
        # Step 4: Remove site-specific boilerplate
        processed_text = self.remove_boilerplate_content(processed_text, site_type)
        
        # Step 5: Clean whitespace
        processed_text = self.clean_whitespace(processed_text)
        
        # Extract structured information
        sentences = self.extract_sentences(processed_text)
        key_phrases = self.extract_key_phrases(processed_text)
        statistics = self.calculate_text_statistics(processed_text)
        
        result = {
            'original_content': content,
            'processed_content': processed_text,
            'sentences': sentences,
            'key_phrases': key_phrases,
            'statistics': statistics,
            'site_type': site_type,
            'processing_metadata': {
                'original_length': len(content),
                'processed_length': len(processed_text),
                'reduction_ratio': 1 - (len(processed_text) / len(content)) if content else 0,
                'sentence_count': len(sentences),
                'key_phrase_count': len(key_phrases)
            }
        }
        
        logger.success(f"Processed content: {len(content)} -> {len(processed_text)} chars "
                      f"({result['processing_metadata']['reduction_ratio']:.2%} reduction)")
        
        return result


# Example usage and testing
if __name__ == "__main__":
    processor = TextProcessor()
    
    # Test with sample content
    sample_content = """
    <h1>Welcome to Our Company</h1>
    <nav>Home | About | Contact</nav>
    <p>This is a sample article about artificial intelligence. 
    AI has revolutionized many industries.</p>
    <p>Machine learning algorithms can process vast amounts of data.</p>
    <div class="advertisement">Ad content here</div>
    <footer>© 2024 Company Name. All rights reserved.</footer>
    """
    
    result = processor.process_content(sample_content, "business")
    
    print("Original content length:", len(sample_content))
    print("Processed content length:", len(result['processed_content']))
    print("Processed content:")
    print(result['processed_content'])
    print("\nSentences:", result['sentences'])
    print("\nKey phrases:", result['key_phrases'])
    print("\nStatistics:", result['statistics'])