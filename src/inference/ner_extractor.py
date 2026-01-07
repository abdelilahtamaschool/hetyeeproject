"""
Rule-based Named Entity Recognition for Dutch legal documents.
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    normalized: Optional[Dict] = None


class DutchLegalNER:
    """
    Rule-based entity extraction for Dutch legal documents.

    Extracts:
    - PERSON: Natural persons (comparanten, erflater, erfgenaam)
    - ORGANIZATION: Legal entities (B.V., N.V., stichting)
    - CADASTRAL: Property identifiers (kadastrale aanduiding)
    - DATE: Legal dates
    - MONEY: Financial amounts
    - ADDRESS: Property addresses
    """

    def __init__(self, use_spacy: bool = False):
        """
        Initialize the NER extractor.

        Args:
            use_spacy: Whether to use spaCy for base NER (requires nl_core_news_lg)
        """
        self.use_spacy = use_spacy
        self.nlp = None

        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("nl_core_news_lg")
            except OSError:
                print("Warning: nl_core_news_lg not found. Install with: python -m spacy download nl_core_news_lg")
                self.use_spacy = False

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for entity extraction."""

        # Cadastral reference patterns
        # e.g., "gemeente Amsterdam, sectie A, nummer 1234"
        # e.g., "kadastraal bekend gemeente Leiden sectie B nummer 5678"
        self.cadastral_pattern = re.compile(
            r'(?:kadastr(?:aal|ale)\s+(?:bekend\s+)?)?'
            r'gemeente\s+([A-Z][a-zA-Z\s\-]+?)[\s,]+'
            r'sectie\s+([A-Z]{1,2})[\s,]+'
            r'(?:nummer|nr\.?)\s+(\d+(?:\s*[a-zA-Z]?\d*)?)',
            re.IGNORECASE
        )

        # Alternative cadastral pattern
        self.cadastral_alt_pattern = re.compile(
            r'([A-Z]{3}\d{2})\s*([A-Z])\s*(\d+)',  # e.g., "ASD01 A 1234"
            re.IGNORECASE
        )

        # Company patterns (B.V., N.V., etc.)
        self.company_pattern = re.compile(
            r'([A-Z][a-zA-Z\s\-&\']+?)\s*'
            r'(B\.?V\.?|N\.?V\.?|C\.?V\.?|V\.?O\.?F\.?|Stichting|Vereniging|Coöperatie)',
            re.IGNORECASE
        )

        # Stichting/Vereniging at start
        self.org_prefix_pattern = re.compile(
            r'(Stichting|Vereniging|Coöperatie)\s+([A-Z][a-zA-Z\s\-&\']+)',
            re.IGNORECASE
        )

        # Date patterns
        # e.g., "1 januari 2024", "01-01-2024", "1/1/2024"
        self.date_patterns = [
            re.compile(
                r'(\d{1,2})\s+'
                r'(januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)\s+'
                r'(\d{4})',
                re.IGNORECASE
            ),
            re.compile(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})'),
            re.compile(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})')
        ]

        # Money patterns
        # e.g., "€ 100.000,00", "EUR 50.000", "honderdduizend euro"
        self.money_pattern = re.compile(
            r'(?:€|EUR|euro)\s*'
            r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            re.IGNORECASE
        )

        # Dutch number words for money
        self.money_words_pattern = re.compile(
            r'(\w+(?:\s+\w+){0,5})\s+'
            r'(?:euro|EUR|€)',
            re.IGNORECASE
        )

        # Person name patterns (Dutch naming conventions)
        # Look for patterns like "de heer/mevrouw [Name]" or "geboren [Name]"
        self.person_patterns = [
            re.compile(
                r'(?:de\s+)?(?:heer|mevrouw|mr\.|mw\.)\s+'
                r'([A-Z][a-zA-Z\s\-\']+(?:\s+(?:van|de|den|der|het|\'t)\s+[A-Z][a-zA-Z\-\']+)?)',
                re.IGNORECASE
            ),
            re.compile(
                r'(?:comparant|erflater|erfgenaam|verkoper|koper|hypotheekgever|hypotheeknemer)(?:e)?\s*'
                r'(?:genaamd|:)?\s*'
                r'([A-Z][a-zA-Z\s\-\']+)',
                re.IGNORECASE
            )
        ]

        # Address pattern
        self.address_pattern = re.compile(
            r'([A-Z][a-zA-Z\s\-]+(?:straat|laan|weg|plein|singel|gracht|kade|dijk))\s*'
            r'(\d+(?:\s*[a-zA-Z])?)\s*'
            r'(?:te|,)?\s*'
            r'(\d{4}\s*[A-Z]{2})?\s*'
            r'([A-Z][a-zA-Z\-]+)?',
            re.IGNORECASE
        )

    def extract_entities(self, text: str) -> Dict[str, List[Entity]]:
        """
        Extract all entities from a legal document.

        Args:
            text: Document text

        Returns:
            Dictionary with entity lists by category
        """
        entities = {
            "subjects": [],        # PERSON entities
            "organizations": [],   # ORGANIZATION entities
            "properties": [],      # CADASTRAL references
            "dates": [],           # DATE entities
            "amounts": [],         # MONEY entities
            "addresses": []        # ADDRESS entities
        }

        # Extract each entity type
        entities["properties"].extend(self._extract_cadastral(text))
        entities["organizations"].extend(self._extract_organizations(text))
        entities["dates"].extend(self._extract_dates(text))
        entities["amounts"].extend(self._extract_money(text))
        entities["subjects"].extend(self._extract_persons(text))
        entities["addresses"].extend(self._extract_addresses(text))

        # Use spaCy for additional entities if available
        if self.use_spacy and self.nlp:
            spacy_entities = self._extract_spacy(text)
            entities = self._merge_entities(entities, spacy_entities)

        # Remove duplicates and sort by position
        for category in entities:
            entities[category] = self._deduplicate_entities(entities[category])

        return entities

    def _extract_cadastral(self, text: str) -> List[Entity]:
        """Extract cadastral references."""
        entities = []

        # Main pattern
        for match in self.cadastral_pattern.finditer(text):
            gemeente = match.group(1).strip()
            sectie = match.group(2).upper()
            nummer = match.group(3).strip()

            entities.append(Entity(
                text=match.group(0),
                label="CADASTRAL",
                start=match.start(),
                end=match.end(),
                normalized={
                    "gemeente": gemeente,
                    "sectie": sectie,
                    "nummer": nummer
                }
            ))

        # Alternative pattern (code format)
        for match in self.cadastral_alt_pattern.finditer(text):
            entities.append(Entity(
                text=match.group(0),
                label="CADASTRAL",
                start=match.start(),
                end=match.end(),
                normalized={
                    "code": match.group(1),
                    "sectie": match.group(2),
                    "nummer": match.group(3)
                }
            ))

        return entities

    def _extract_organizations(self, text: str) -> List[Entity]:
        """Extract organization names."""
        entities = []

        # Companies with suffix (B.V., N.V., etc.)
        for match in self.company_pattern.finditer(text):
            name = match.group(1).strip()
            org_type = match.group(2)

            # Skip if name is too short or just common words
            if len(name) < 2 or name.lower() in ['de', 'het', 'een']:
                continue

            entities.append(Entity(
                text=match.group(0),
                label="ORGANIZATION",
                start=match.start(),
                end=match.end(),
                normalized={
                    "name": name,
                    "type": org_type
                }
            ))

        # Organizations with prefix (Stichting, Vereniging)
        for match in self.org_prefix_pattern.finditer(text):
            entities.append(Entity(
                text=match.group(0),
                label="ORGANIZATION",
                start=match.start(),
                end=match.end(),
                normalized={
                    "type": match.group(1),
                    "name": match.group(2).strip()
                }
            ))

        return entities

    def _extract_dates(self, text: str) -> List[Entity]:
        """Extract dates."""
        entities = []

        for pattern in self.date_patterns:
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(0),
                    label="DATE",
                    start=match.start(),
                    end=match.end()
                ))

        return entities

    def _extract_money(self, text: str) -> List[Entity]:
        """Extract monetary amounts."""
        entities = []

        for match in self.money_pattern.finditer(text):
            amount = match.group(1).replace('.', '').replace(',', '.')

            entities.append(Entity(
                text=match.group(0),
                label="MONEY",
                start=match.start(),
                end=match.end(),
                normalized={
                    "amount": amount,
                    "currency": "EUR"
                }
            ))

        return entities

    def _extract_persons(self, text: str) -> List[Entity]:
        """Extract person names."""
        entities = []

        for pattern in self.person_patterns:
            for match in pattern.finditer(text):
                name = match.group(1).strip() if match.lastindex >= 1 else match.group(0)

                # Clean up the name
                name = re.sub(r'\s+', ' ', name).strip()

                # Skip if too short or common words
                if len(name) < 3:
                    continue

                entities.append(Entity(
                    text=name,
                    label="PERSON",
                    start=match.start(),
                    end=match.end()
                ))

        return entities

    def _extract_addresses(self, text: str) -> List[Entity]:
        """Extract addresses."""
        entities = []

        for match in self.address_pattern.finditer(text):
            street = match.group(1)
            number = match.group(2) if match.group(2) else ""
            postcode = match.group(3) if match.group(3) else ""
            city = match.group(4) if match.group(4) else ""

            entities.append(Entity(
                text=match.group(0),
                label="ADDRESS",
                start=match.start(),
                end=match.end(),
                normalized={
                    "street": street.strip(),
                    "number": number.strip(),
                    "postcode": postcode.strip(),
                    "city": city.strip()
                }
            ))

        return entities

    def _extract_spacy(self, text: str) -> Dict[str, List[Entity]]:
        """Use spaCy for additional entity extraction."""
        entities = {
            "subjects": [],
            "organizations": [],
            "properties": [],
            "dates": [],
            "amounts": [],
            "addresses": []
        }

        doc = self.nlp(text)

        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.8  # Lower confidence for spaCy entities
            )

            if ent.label_ == "PERSON":
                entities["subjects"].append(entity)
            elif ent.label_ in ["ORG", "ORGANIZATION"]:
                entities["organizations"].append(entity)
            elif ent.label_ == "DATE":
                entities["dates"].append(entity)
            elif ent.label_ == "MONEY":
                entities["amounts"].append(entity)
            elif ent.label_ in ["LOC", "GPE"]:
                entities["addresses"].append(entity)

        return entities

    def _merge_entities(
        self,
        rule_entities: Dict[str, List[Entity]],
        spacy_entities: Dict[str, List[Entity]]
    ) -> Dict[str, List[Entity]]:
        """Merge rule-based and spaCy entities, preferring rule-based."""
        merged = {k: list(v) for k, v in rule_entities.items()}

        for category, ents in spacy_entities.items():
            for ent in ents:
                # Check for overlap with existing entities
                overlaps = False
                for existing in merged[category]:
                    if self._entities_overlap(ent, existing):
                        overlaps = True
                        break

                if not overlaps:
                    merged[category].append(ent)

        return merged

    def _entities_overlap(self, e1: Entity, e2: Entity) -> bool:
        """Check if two entities overlap."""
        return not (e1.end <= e2.start or e2.end <= e1.start)

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping the one with highest confidence."""
        if not entities:
            return []

        # Sort by start position
        sorted_ents = sorted(entities, key=lambda e: (e.start, -e.confidence))

        deduplicated = []
        for ent in sorted_ents:
            # Check if overlaps with any existing
            overlaps = False
            for existing in deduplicated:
                if self._entities_overlap(ent, existing):
                    overlaps = True
                    # Keep the higher confidence one
                    if ent.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(ent)
                    break

            if not overlaps:
                deduplicated.append(ent)

        return sorted(deduplicated, key=lambda e: e.start)

    def to_dict(self, entities: Dict[str, List[Entity]]) -> Dict:
        """
        Convert entities to JSON-serializable dictionary.

        Args:
            entities: Dictionary of entity lists

        Returns:
            JSON-serializable dictionary
        """
        result = {}

        for category, ents in entities.items():
            result[category] = [
                {
                    "text": e.text,
                    "label": e.label,
                    "start": e.start,
                    "end": e.end,
                    "confidence": e.confidence,
                    "normalized": e.normalized
                }
                for e in ents
            ]

        return result
