import logging
import re
import time
from typing import List, Dict
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from discord.ext import commands
from googlesearch import search

from common.llm.classes import Tool, ToolCall, ToolResponseMessage, MessageGroup

logger = logging.getLogger(f'MARIA3.{__name__.split(".")[-1]}')

# CONSTANTES ----------------------------------------------------

DEFAULT_CHUNK_SIZE = 2500
DEFAULT_NUM_RESULTS = 5
DEFAULT_TIMEOUT = 15
CACHE_EXPIRY_HOURS = 24

# Headers pour éviter les blocages
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Patterns pour nettoyer le contenu
NOISE_PATTERNS = [
    r'En poursuivant votre navigation.*?cookies.*?\.',
    r'This site uses cookies.*?privacy.*?\.',
    r'Nous utilisons des cookies.*?confidentialité.*?\.',
    r'Accept [aA]ll cookies',
    r'cookie policy',
    r'privacy policy',
    r'terms of service',
]

# Sélecteurs pour détecter le contenu principal
MAIN_CONTENT_SELECTORS = [
    'main', 'article', '.content', '.post', '.post-content', '.entry-content',
    '#content', '#main', '.article', '[role="main"]'
]

class Web(commands.Cog):
    """Cog pour les outils de recherche et navigation web utilisés par l'IA."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.page_chunks_cache: Dict[str, Dict] = {}
        
        # Outils globaux exportés
        self.GLOBAL_TOOLS = [
            Tool(
                name='search_web_pages',
                description='Recherche des pages web et renvoie une liste des pages trouvées avec titre, URL et description.',
                properties={
                    'query': {'type': 'string', 'description': 'Requête de recherche'},
                    'num_results': {'type': 'integer', 'description': 'Nombre de résultats (max 10)'},
                    'lang': {'type': 'string', 'description': "Code de langue ('fr', 'en', etc.)"}
                },
                function=self._tool_search_web_pages
            ),
            Tool(
                name='read_web_page',
                description='Lit le contenu d\'une page web et retourne une partie spécifique.',
                properties={
                    'url': {'type': 'string', 'description': 'URL de la page web à lire'},
                    'chunk_index': {'type': 'integer', 'description': 'Index de la partie à lire (défaut: 0)'}
                },
                function=self._tool_read_web_page
            )
        ]
    
    # MÉTHODES UTILITAIRES --------------------------------------------
    
    def search_web_pages(self, query: str, lang: str = 'fr', num_results: int = DEFAULT_NUM_RESULTS) -> List[Dict]:
        """Effectue une recherche web."""
        try:
            results = search(query, lang=lang, num_results=min(num_results, 10), advanced=True, safe='off')
            return [{'title': r.title, 'url': r.url, 'description': r.description} for r in results]
        except Exception as e:
            logger.error(f"Erreur recherche web: {e}")
            return []
    
    def fetch_page_chunks(self, url: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
        """Récupère et divise le contenu d'une page web."""
        # Vérifier le cache
        cache_key = f"{url}_{chunk_size}"
        if cache_key in self.page_chunks_cache:
            cache_entry = self.page_chunks_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < CACHE_EXPIRY_HOURS * 3600:
                return cache_entry['chunks']
        
        try:
            # Récupération de la page
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=DEFAULT_TIMEOUT)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Suppression des éléments non pertinents
            for tag in soup(["script", "style", "header", "footer", "nav", "aside", "iframe", 
                            "noscript", "form", "button", "svg", ".ad", ".ads", ".cookie", 
                            ".popup", ".banner", ".sidebar", ".menu", ".comments"]):
                tag.decompose()
            
            # Détection du contenu principal
            main_content = None
            for selector in MAIN_CONTENT_SELECTORS:
                content = soup.select(selector)
                if content and len(str(content[0])) > 500:
                    main_content = content[0]
                    break
            
            text_container = main_content or soup.find('body') or soup
            
            # Extraction du texte
            text = ""
            for elem in text_container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                content = elem.get_text(strip=True)
                if len(content) > 20 or (elem.name.startswith('h') and content):
                    prefix = f"## " if elem.name.startswith('h') else ""
                    text += f"\n{prefix}{content}\n"
            
            # Nettoyage du texte
            for pattern in NOISE_PATTERNS:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)
            text = text.strip()
            
            # Division en chunks
            chunks = []
            paragraphs = [p for p in re.split(r'\n\n+', text) if p.strip()]
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) + 2 > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    current_chunk = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Filtrer les chunks trop courts et mettre en cache
            chunks = [c for c in chunks if len(c.strip()) > 100]
            self.page_chunks_cache[cache_key] = {
                'chunks': chunks, 
                'timestamp': time.time()
            }
            
            return chunks
            
        except Exception as e:
            logger.error(f"Erreur lecture page {url}: {e}")
            return []
    
    # OUTILS ----------------------------------------------------------
    
    def _tool_search_web_pages(self, tool_call: ToolCall, context: MessageGroup) -> ToolResponseMessage:
        """Outil pour rechercher des pages web."""
        query = tool_call.arguments.get('query')
        if not query:
            return ToolResponseMessage({'error': 'Requête manquante'}, tool_call.data['id'])
        
        num_results = tool_call.arguments.get('num_results', DEFAULT_NUM_RESULTS)
        lang = tool_call.arguments.get('lang', 'fr')
        
        results = self.search_web_pages(query, lang, num_results)
        
        if not results:
            return ToolResponseMessage({'error': 'Aucun résultat trouvé'}, tool_call.data['id'])
        
        return ToolResponseMessage({
            'query': query,
            'results': results,
            'total_results': len(results)
        }, tool_call.data['id'], header=f'Recherche web pour *\"{query}\"*')
    
    def _tool_read_web_page(self, tool_call: ToolCall, context: MessageGroup) -> ToolResponseMessage:
        """Outil pour lire le contenu d'une page web."""
        url = tool_call.arguments.get('url')
        if not url:
            return ToolResponseMessage({'error': 'URL manquante'}, tool_call.data['id'])
        
        # Validation basique de l'URL
        parsed = urlparse(url)
        if parsed.scheme not in ['http', 'https'] or not parsed.netloc:
            return ToolResponseMessage({'error': 'URL invalide'}, tool_call.data['id'])
        
        chunk_index = tool_call.arguments.get('chunk_index', 0)
        chunks = self.fetch_page_chunks(url)
        
        if not chunks:
            return ToolResponseMessage({'error': 'Impossible de lire cette page'}, tool_call.data['id'])
        
        if chunk_index >= len(chunks):
            return ToolResponseMessage({
                'error': f'Index {chunk_index} hors limites',
                'total_chunks': len(chunks)
            }, tool_call.data['id'])
            
        header_url = f'[{url.split("//")[-1].split("/")[0]}](<{url}>)'
        return ToolResponseMessage({
            'url': url,
            'chunk_index': chunk_index,
            'total_chunks': len(chunks),
            'content': chunks[chunk_index],
            'has_more': chunk_index < len(chunks) - 1
        }, tool_call.data['id'], header=f'Lecture de {header_url}')

async def setup(bot):
    await bot.add_cog(Web(bot))
