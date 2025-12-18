# ADR 5: Multilingual Support System

## Status
Accepted

## Date
2025-12-15

## Context
The textbook aims to be accessible to a diverse audience, including Urdu-speaking students and educators. The multilingual support system must provide accurate translations while maintaining the technical accuracy and educational value of the content. The system should be easy to implement, maintain, and extend to other languages in the future, while respecting the project's constraints on content complexity and accuracy.

## Decision
We will implement a client-side toggle system for Urdu translation with the following characteristics:

**Client-Side Implementation**:
- Language toggle button/component on each chapter page
- Switches between English and Urdu content on the same page
- Uses React state management for immediate language switching
- Maintains navigation structure and page layout
- Preserves technical content structure while translating narrative text

**Content Handling**:
- Technical elements (code blocks, diagrams, mathematical formulas) remain in English
- Narrative text, explanations, and descriptions translated to Urdu
- Maintains chapter structure and section organization
- Preserves all code examples and technical references in English
- Supports progressive translation (partial content can be translated)

**Implementation Approach**:
- Urdu content stored alongside English content in same chapter files
- JavaScript-based toggle functionality for language switching
- Preserves all metadata, citations, and references in both languages
- Supports mixed-language scenarios where appropriate

## Alternatives Considered
1. **Server-side rendering**: Better SEO but more complex implementation
2. **Separate language sites**: Complete separation but harder to maintain
3. **URL-based routing**: /en/, /ur/ paths but affects navigation and user experience
4. **Static generation**: Build separate sites but increases build complexity and maintenance
5. **Dynamic loading**: Load translations via API but adds network dependency

## Consequences
**Positive:**
- Simple implementation without complex routing
- Maintains content structure and navigation
- Allows for progressive translation (partial content)
- Good user experience with immediate language switching
- Easy to extend to other languages
- Maintains technical accuracy by keeping code/technical terms in English
- Reduces maintenance overhead compared to separate sites

**Negative:**
- Larger bundle sizes due to dual language content
- SEO implications for multilingual content
- Requires careful coordination between translations
- May complicate content editing workflow
- Limited to languages with similar document structure

## References
- plan.md: Bonus Features section (Urdu translation toggle)
- research.md: Translation System Research section
- data-model.md: Chapter entity with urdu_content field
- spec.md: Personalization entry point and Urdu translation toggle requirements