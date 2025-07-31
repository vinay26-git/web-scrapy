"""
Advanced text preprocessing and cleaning module.
Implements enterprise-grade text normalization, cleaning, and quality assessment.
"""
import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from langdetect import detect, DetectorFactory
import textstat
from loguru import logger

from .scraper import ScrapedContent
from .config import Config

# Set seed for consistent language detection
DetectorFactory.seed = 0


@dataclass
class ProcessedContent:
    """Container for processed content with quality metrics."""
    original_content: ScrapedContent
    cleaned_text: str
    sentences: List[str]
    quality_metrics: Dict
    language: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextPreprocessor:
    """
    Enterprise-grade text preprocessor with advanced cleaning and normalization.
    
    Features:
    - Unicode normalization and encoding handling
    - Language detection and filtering
    - Advanced text cleaning and denoising
    - Quality assessment and scoring
    - Sentence segmentation and tokenization
    - Readability analysis
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_nltk()
        self._setup_patterns()
    
    def _setup_nltk(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
    
    def _setup_patterns(self):
        """Setup regex patterns for text cleaning."""
        self.patterns = {
            # URLs and email addresses
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # Social media handles
            'twitter_handle': re.compile(r'@[A-Za-z0-9_]+'),
            'hashtag': re.compile(r'#[A-Za-z0-9_]+'),
            
            # HTML entities and tags (backup cleanup)
            'html_entity': re.compile(r'&[a-zA-Z0-9#]+;'),
            'html_tag': re.compile(r'<[^>]+>'),
            
            # Special characters and symbols
            'multiple_spaces': re.compile(r'\s{2,}'),
            'multiple_newlines': re.compile(r'\n{3,}'),
            'bullet_points': re.compile(r'^[\s]*[•·▪▫‣⁃]\s*', re.MULTILINE),
            'dashes': re.compile(r'[—–−‒]'),
            'quotes': re.compile(r'[""''‚„]'),
            
            # Numbers and dates
            'phone_number': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'date_pattern': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            
            # Currency and financial
            'currency': re.compile(r'[$€£¥¢]\s*\d+(?:,\d{3})*(?:\.\d{2})?'),
            
            # Common noise patterns
            'copyright': re.compile(r'©.*?\d{4}.*?(?:\.|$)', re.IGNORECASE),
            'all_rights_reserved': re.compile(r'all rights reserved.*?(?:\.|$)', re.IGNORECASE),
            'powered_by': re.compile(r'powered by.*?(?:\.|$)', re.IGNORECASE),
            
            # Navigation and UI elements
            'breadcrumb': re.compile(r'home\s*[>›]\s*.*?[>›]', re.IGNORECASE),
            'click_here': re.compile(r'click here.*?(?:\.|$)', re.IGNORECASE),
            'read_more': re.compile(r'read more.*?(?:\.|$)', re.IGNORECASE),
        }
    
    def _detect_language(self, text: str) -> str:
        """Detect text language using langdetect."""
        try:
            # Use first 1000 chars for detection (faster and usually accurate)
            sample_text = text[:1000] if len(text) > 1000 else text
            language = detect(sample_text)
            return language
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"  # Default to English
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters and handle encoding issues."""
        # Normalize Unicode to NFKC form
        text = unicodedata.normalize('NFKC', text)
        
        # Handle common encoding issues
        replacements = {
            'â€™': "'",  # Right single quotation mark
            'â€œ': '"',  # Left double quotation mark
            'â€�': '"',  # Right double quotation mark
            'â€"': '—',  # Em dash
            'â€"': '–',  # En dash
            'â€¦': '...',  # Horizontal ellipsis
            'Â ': ' ',   # Non-breaking space
            'Â': '',     # Misc encoding artifacts
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace."""
        # Replace multiple spaces with single space
        text = self.patterns['multiple_spaces'].sub(' ', text)
        
        # Replace multiple newlines with double newline
        text = self.patterns['multiple_newlines'].sub('\n\n', text)
        
        # Clean up line breaks and spaces
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _remove_noise_patterns(self, text: str) -> str:
        """Remove common noise patterns from text."""
        # Remove URLs and email addresses
        text = self.patterns['url'].sub('', text)
        text = self.patterns['email'].sub('', text)
        
        # Remove social media elements
        text = self.patterns['twitter_handle'].sub('', text)
        text = self.patterns['hashtag'].sub('', text)
        
        # Remove HTML remnants
        text = self.patterns['html_entity'].sub('', text)
        text = self.patterns['html_tag'].sub('', text)
        
        # Remove common noise patterns
        text = self.patterns['copyright'].sub('', text)
        text = self.patterns['all_rights_reserved'].sub('', text)
        text = self.patterns['powered_by'].sub('', text)
        text = self.patterns['breadcrumb'].sub('', text)
        text = self.patterns['click_here'].sub('', text)
        text = self.patterns['read_more'].sub('', text)
        
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation and special characters."""
        # Normalize dashes
        text = self.patterns['dashes'].sub('—', text)
        
        # Normalize quotes
        text = self.patterns['quotes'].sub('"', text)
        
        # Clean bullet points
        text = self.patterns['bullet_points'].sub('• ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])\s*([a-zA-Z])', r'\1 \2', text)  # Add space after punctuation
        
        return text
    
    def _segment_sentences(self, text: str, language: str = 'english') -> List[str]:
        """Segment text into sentences using NLTK."""
        try:
            # Use language-specific sentence tokenizer if available
            lang_map = {
                'en': 'english',
                'es': 'spanish',
                'fr': 'french',
                'de': 'german',
                'it': 'italian',
                'pt': 'portuguese',
                'nl': 'dutch',
                'ru': 'russian',
            }
            
            nltk_language = lang_map.get(language, 'english')
            sentences = sent_tokenize(text, language=nltk_language)
            
            # Filter out very short or very long sentences
            filtered_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if 10 <= len(sentence) <= 1000:  # Reasonable sentence length
                    filtered_sentences.append(sentence)
            
            return filtered_sentences
            
        except Exception as e:
            logger.warning(f"Sentence segmentation failed: {e}")
            # Fallback to simple splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _calculate_quality_metrics(self, text: str, sentences: List[str]) -> Dict:
        """Calculate comprehensive quality metrics for the text."""
        metrics = {}
        
        # Basic statistics
        metrics['character_count'] = len(text)
        metrics['word_count'] = len(text.split())
        metrics['sentence_count'] = len(sentences)
        metrics['avg_sentence_length'] = metrics['word_count'] / max(metrics['sentence_count'], 1)
        
        # Readability scores
        try:
            metrics['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            metrics['flesch_kincaid_grade'] = textstat.flesch_kincaid().score(text)
            metrics['automated_readability_index'] = textstat.automated_readability_index(text)
            metrics['coleman_liau_index'] = textstat.coleman_liau_index(text)
        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            metrics.update({
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'automated_readability_index': 0,
                'coleman_liau_index': 0
            })
        
        # Text complexity metrics
        unique_words = len(set(text.lower().split()))
        metrics['lexical_diversity'] = unique_words / max(metrics['word_count'], 1)
        
        # Content quality indicators
        metrics['avg_word_length'] = sum(len(word) for word in text.split()) / max(metrics['word_count'], 1)
        metrics['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        metrics['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        metrics['punctuation_ratio'] = sum(1 for c in text if c in '.,!?;:') / max(len(text), 1)
        
        # Calculate overall quality score (0-100)
        quality_score = self._calculate_overall_quality(metrics)
        metrics['overall_quality_score'] = quality_score
        
        return metrics
    
    def _calculate_overall_quality(self, metrics: Dict) -> float:
        """Calculate an overall quality score based on various metrics."""
        score = 100.0
        
        # Penalize very short or very long content
        word_count = metrics['word_count']
        if word_count < 50:
            score -= 30
        elif word_count < 100:
            score -= 15
        elif word_count > 5000:
            score -= 10
        
        # Penalize poor readability
        flesch_score = metrics.get('flesch_reading_ease', 50)
        if flesch_score < 30:  # Very difficult
            score -= 20
        elif flesch_score < 50:  # Difficult
            score -= 10
        
        # Penalize low lexical diversity
        diversity = metrics['lexical_diversity']
        if diversity < 0.3:
            score -= 15
        elif diversity < 0.5:
            score -= 5
        
        # Penalize excessive uppercase (spam-like)
        if metrics['uppercase_ratio'] > 0.1:
            score -= 20
        
        # Penalize excessive digits (spam-like)
        if metrics['digit_ratio'] > 0.2:
            score -= 15
        
        return max(0, min(100, score))
    
    def process_content(self, scraped_content: ScrapedContent) -> Optional[ProcessedContent]:
        """
        Process scraped content with comprehensive cleaning and analysis.
        
        Args:
            scraped_content: ScrapedContent object to process
            
        Returns:
            ProcessedContent object or None if processing failed
        """
        try:
            logger.info(f"Processing content from {scraped_content.url}")
            
            text = scraped_content.content
            
            # Step 1: Unicode normalization
            text = self._normalize_unicode(text)
            
            # Step 2: Remove noise patterns
            text = self._remove_noise_patterns(text)
            
            # Step 3: Normalize punctuation
            text = self._normalize_punctuation(text)
            
            # Step 4: Clean whitespace
            text = self._clean_whitespace(text)
            
            # Step 5: Detect language
            language = self._detect_language(text)
            
            # Step 6: Sentence segmentation
            sentences = self._segment_sentences(text, language)
            
            # Step 7: Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(text, sentences)
            
            # Create processed content object
            processed_content = ProcessedContent(
                original_content=scraped_content,
                cleaned_text=text,
                sentences=sentences,
                quality_metrics=quality_metrics,
                language=language,
                metadata={
                    'processing_steps': [
                        'unicode_normalization',
                        'noise_removal',
                        'punctuation_normalization',
                        'whitespace_cleaning',
                        'language_detection',
                        'sentence_segmentation',
                        'quality_assessment'
                    ],
                    'original_length': len(scraped_content.content),
                    'cleaned_length': len(text),
                    'compression_ratio': len(text) / max(len(scraped_content.content), 1)
                }
            )
            
            logger.success(f"Successfully processed content from {scraped_content.url}")
            logger.info(f"Quality score: {quality_metrics['overall_quality_score']:.1f}/100")
            
            return processed_content
            
        except Exception as e:
            logger.error(f"Failed to process content from {scraped_content.url}: {e}")
            return None
    
    def process_batch(self, scraped_contents: List[ScrapedContent]) -> List[ProcessedContent]:
        """
        Process multiple scraped contents in batch.
        
        Args:
            scraped_contents: List of ScrapedContent objects
            
        Returns:
            List of successfully processed contents
        """
        processed_contents = []
        
        for i, content in enumerate(scraped_contents):
            logger.info(f"Processing content {i+1}/{len(scraped_contents)}")
            
            processed = self.process_content(content)
            if processed:
                processed_contents.append(processed)
        
        logger.info(f"Successfully processed {len(processed_contents)}/{len(scraped_contents)} contents")
        return processed_contents
    
    def filter_by_quality(self, processed_contents: List[ProcessedContent], 
                         min_quality_score: float = 50.0) -> List[ProcessedContent]:
        """
        Filter processed contents by quality score.
        
        Args:
            processed_contents: List of ProcessedContent objects
            min_quality_score: Minimum quality score threshold
            
        Returns:
            Filtered list of high-quality contents
        """
        high_quality_contents = []
        
        for content in processed_contents:
            quality_score = content.quality_metrics['overall_quality_score']
            if quality_score >= min_quality_score:
                high_quality_contents.append(content)
            else:
                logger.debug(f"Filtered out low-quality content: {quality_score:.1f}/100")
        
        logger.info(f"Kept {len(high_quality_contents)}/{len(processed_contents)} high-quality contents")
        return high_quality_contents