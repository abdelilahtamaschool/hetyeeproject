### uitvoering
Taak: Update model.py naar ModernBERT voor verbeterde performance en een context window tot 8192 tokens.

Context: > * Gebruik de context uit project_plan.md (rekening houdend met de specifieke projectdoelen).

Hardware optimalisatie: De code draait op een RTX 4070, dus maak gebruik van GPU-acceleratie en Flash Attention 2 indien mogelijk.

Modelkeuze: Gebruik answerdotai/ModernBERT-base (of -large).

Vereisten voor de code:

Vervang de standaard BERT-architectuur door ModernBERT via de Hugging Face transformers library.

Zorg dat de tokenizer en het model correct zijn geconfigureerd voor langere sequences (>512 tokens).

Voeg device-handling toe (cuda voor de RTX 4070).

Houd de interface van model.py consistent met de rest van de bestaande applicatiestructuur.

Documentatie van Stappen (t.b.v. Eindrapport)
Sla de onderstaande sectie op als een apart Markdown-bestand (bijv. ontwikkelstappen_modernbert.md).

Documentatie Ontwikkelstappen: Transitie naar ModernBERT
1. Analyse Projectcontext
Er is een grondige analyse uitgevoerd van project_plan.md om te waarborgen dat de modelkeuze aansluit bij de functionele eisen van het project. De focus ligt op het verwerken van documenten die de standaard limiet van 512 tokens overschrijden.

2. Hardware-specificatie & Optimalisatie
Er is vastgesteld dat de hardware (NVIDIA RTX 4070) voldoende VRAM en rekenkracht biedt voor ModernBERT.

Target Device: CUDA-versnelling wordt geforceerd in de implementatie.

Efficiëntie: Gebruik van Flash Attention wordt voorbereid om de verwerking van lange teksten te versnellen.

3. Modelselectie
De keuze is gevallen op ModernBERT van AnswerDotAI. De belangrijkste motivaties hiervoor zijn:

Context Window: Verhoging van de limiet van 512 naar 8192 tokens.

Architectuur: Gebruik van 'unpadding' en modernere activatiefuncties waardoor het model sneller en nauwkeuriger is dan de originele BERT.

4. Herstructurering model.py
De volgende technische wijzigingen zijn gepland voor de broncode:

Verwijderen van verouderde BertModel klassen.

Implementatie van de AutoModel klasse met ondersteuning voor de ModernBERT gewichten.

Aanpassing van de tokenizer om gebruik te maken van de uitgebreide padding- en truncation-instellingen voor langere inputs.

5. Validatie
Na implementatie wordt het model getest op:

Correcte herkenning door de GPU.

Succesvolle verwerking van een input sequence van >1024 tokens om de stabiliteit van het context window te bevestigen.