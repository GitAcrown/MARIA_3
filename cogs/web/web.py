import logging
from typing import Optional, Literal
from pathlib import Path
import re
import base64
from datetime import datetime
from googlesearch import search as google_search
from playwright.async_api import async_playwright, Browser, Page, Playwright
from discord.ext import commands

from common.llm.classes import Tool, ToolCall, MessageGroup, ToolResponseMessage

logger = logging.getLogger(f'MARIA3.{__name__.split(".")[-1]}')

# CONSTANTS -----------------------------------------------------------------

TEMP_DIR = Path('./temp')
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Durée de conservation des screenshots (en heures)
SCREENSHOT_MAX_AGE_HOURS = 24
WEB_CHUNK_SIZE = 1000

# EXCEPTIONS -----------------------------------------------------------------

class WebAgentError(Exception):
    pass

class WebRequestError(WebAgentError):
    pass

class WebResponseError(WebAgentError):
    pass

# COG =======================================================================
class Web(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

        self.timeout = 15000
        self.viewport_width = 1280
        self.viewport_height = 720

        self._playwright = None
        self._browser = None
        self._page = None
        
        # Cache pour le contenu des pages web
        self.__web_pages = {}
        
        # Compteur pour le nettoyage automatique
        self._cleanup_counter = 0
        self._cleanup_frequency = 10  # Nettoie tous les 10 screenshots

        # Outils
        self.AGENT_TOOLS = [
            Tool(
                name='search_web_pages',
                description="Recherche des informations sur le web.",
                properties={
                    'query': {
                        'type': 'string',
                        'description': "La requête de recherche"
                    },
                    'lang': {
                        'type': 'string',
                        'description': "La langue de la recherche ('fr', 'en'...)"
                    },
                    'num_results': {
                        'type': 'integer',
                        'description': "Le nombre de résultats à retourner (max. 10)"
                    }
                },
                function=self._tool_search_web_pages
            ),
            Tool(
                name='get_webpage_content_chunk',
                description="Récupère un bloc de contenu texte d'une page web. A utiliser itérativement pour lire le contenu d'une page web renvoyée par l'outil de recherche web.",
                properties={
                    'url': {
                        'type': 'string',
                        'description': "L'URL de la page à lire"
                    },
                    'index': {
                        'type': 'integer',
                        'description': "L'index du bloc à récupérer (0-indexé)"
                    },
                    'javascript': {
                        'type': 'boolean',
                        'description': "Si True, active JavaScript pour la page web (par défaut: False)"
                    }
                },
                function=self._tool_get_webpage_content_chunk
            )
        ]
    
    async def cog_unload(self):
        """Nettoie les ressources lors du déchargement du cog."""
        await self._cleanup_browser()

    # Outils -------------------------------------------------------------------

    def _tool_search_web_pages(self, tool_call: ToolCall, context: MessageGroup) -> ToolResponseMessage:
        query = tool_call.arguments.get('query')
        num = tool_call.arguments.get('num_results', 5)
        lang = tool_call.arguments.get('lang', 'fr')
        if not query:
            return ToolResponseMessage({'error': 'Aucune requête fournie.'}, tool_call.data['id'])
        
        try:
            results = []
            for r in self.search_web_pages(query, lang, num):
                results.append({'title': r.title, 'url': r.url, 'description': r.description}) #type: ignore
            return ToolResponseMessage({'results': results}, tool_call.data['id'], header=f"**Recherche web pour** '*{query}*'")
        except Exception as e:
            return ToolResponseMessage({'error': str(e)}, tool_call.data['id'])
        
    async def _tool_get_webpage_content_chunk(self, tool_call: ToolCall, context: MessageGroup) -> ToolResponseMessage:
        url = tool_call.arguments.get('url')
        if not url:
            return ToolResponseMessage({'error': 'Aucune URL fournie.'}, tool_call.data['id'])
        
        index = tool_call.arguments.get('index', 0)
        if not isinstance(index, int) or index < 0:
            return ToolResponseMessage({'error': 'Index invalide.', 'chunk_index': index, 'total_chunks': 0}, tool_call.data['id'])
            
        js = tool_call.arguments.get('javascript', False)
        if url in self.__web_pages:
            content = self.__web_pages[url]
        else:
            try:
                content = await self.extract_text_content(url, enable_js=js)
                self.__web_pages[url] = content
            except Exception as e:
                return ToolResponseMessage({'error': str(e), 'chunk_index': index, 'total_chunks': 0}, tool_call.data['id'])
        
        # On coupe par blocs de WEB_CHUNK_SIZE en essayant de pas couper de mot en deux
        chunks = []
        current_chunk = ''
        for line in content.split('\n'):
            if len(current_chunk) + len(line) <= WEB_CHUNK_SIZE:
                current_chunk += line + '\n'
            else:
                chunks.append(current_chunk)
                current_chunk = line + '\n'
        if current_chunk:
            chunks.append(current_chunk)
            
        if index >= len(chunks):
            return ToolResponseMessage({'error': 'Index hors limites.', 'chunk_index': index, 'total_chunks': len(chunks)}, tool_call.data['id'])
        
        # On affiche dans le header que le nom de domaine mais on markdown l'url
        header_url = f'[{url.split("//")[-1].split("/")[0]}](<{url}>)'
        return ToolResponseMessage({'content': chunks[index], 'chunk_index': index, 'total_chunks': len(chunks)}, tool_call.data['id'], header=f"**Consultation de** {header_url}")
    
    # GESTION DU NAVIGATEUR -----------------------------------------------------
    
    async def _get_browser(self) -> tuple[Browser, Page]:
        """Crée ou réutilise un navigateur et une page avec les paramètres optimisés."""
        if self._browser and self._page:
            return self._browser, self._page
        
        # Fermer l'ancienne instance si elle existe
        await self._cleanup_browser()
        
        # Créer une nouvelle instance optimisée
        if not self._playwright:
            self._playwright = await async_playwright().start()
        
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-extensions',
                '--disable-plugins',
                '--disable-images',  # Désactiver les images pour le texte uniquement
                '--disable-javascript',  # Plus rapide si pas besoin de JS
                '--disable-gpu',
                '--no-first-run',
                '--no-default-browser-check',
                '--disable-default-apps',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding'
            ]
        )
        
        self._page = await self._browser.new_page(
            viewport={'width': self.viewport_width, 'height': self.viewport_height}
        )
        self._page.set_default_timeout(self.timeout)
        
        # Désactiver les ressources inutiles pour plus de vitesse
        await self._page.route("**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2}", lambda route: route.abort())
        
        return self._browser, self._page
    
    async def _cleanup_browser(self):
        """Nettoie les ressources du navigateur."""
        if self._page:
            await self._page.close()
            self._page = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
    
    # NETTOYAGE DES FICHIERS ----------------------------------------------------
    
    def cleanup_old_screenshots(self, max_age_hours: int = SCREENSHOT_MAX_AGE_HOURS) -> int:
        """
        Nettoie les anciens screenshots du dossier temporaire.
        
        Args:
            max_age_hours: Âge maximum des fichiers en heures avant suppression
            
        Returns:
            Nombre de fichiers supprimés
        """
        if not TEMP_DIR.exists():
            return 0
        
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        deleted_count = 0
        
        try:
            for file_path in TEMP_DIR.glob("ss_*.png"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
        except Exception:
            # Ignore les erreurs de suppression (fichier déjà supprimé, permissions, etc.)
            pass
        
        return deleted_count
    
    # RECHERCHE WEB -------------------------------------------------------------

    def search_web_pages(self, query: str, lang: str = 'fr', num_results: int = 5):
        """Recherche des informations sur le web."""
        results = google_search(query, lang=lang, num_results=num_results, advanced=True, safe='off')
        return results
    
    # EXTRACTION DE CONTENU WEB -------------------------------------------------
    
    async def extract_text_content(self, url: str, remove_navigation: bool = True, enable_js: bool = False) -> str:
        """
        Extrait le contenu textuel principal d'une page web (optimisé pour la vitesse).
        
        Args:
            url: L'URL de la page à analyser
            remove_navigation: Si True, supprime les éléments de navigation
            enable_js: Si True, active JavaScript (plus lent mais parfois nécessaire)
            
        Returns:
            Le contenu textuel nettoyé et lisible par une IA
        """
        try:
            browser, page = await self._get_browser()
            
            # Si JS est nécessaire, on réactive temporairement
            if enable_js:
                await page.set_extra_http_headers({'Cache-Control': 'no-cache'})
                await page.context.clear_cookies()
            
            # Navigation rapide
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=self.timeout)
                # Attente réduite
                await page.wait_for_timeout(500 if not enable_js else 1500)
            except Exception as e:
                raise WebRequestError(f"Navigation impossible vers {url}: {str(e)}")
            
            try:
                # Extraction optimisée du contenu
                main_content = await page.evaluate("""
                    (removeNavigation) => {
                        // Suppression rapide des éléments indésirables
                        if (removeNavigation) {
                            const toRemove = document.querySelectorAll('nav,header,footer,aside,.nav,.menu,.ads,.ad,.sidebar,.cookie-notice,script,style,noscript');
                            toRemove.forEach(el => el.remove());
                        }
                        
                        // Extraction prioritaire
                        const selectors = ['main', 'article', '.content', '#content', '.main-content', '.post-content', '.entry-content', '[role="main"]'];
                        
                        for (const selector of selectors) {
                            const element = document.querySelector(selector);
                            if (element && element.innerText.trim().length > 100) {
                                return element.innerText;
                            }
                        }
                        
                        // Fallback sur body
                        return document.body.innerText || document.body.textContent || '';
                    }
                """, remove_navigation)
                
                cleaned_text = self._clean_text(main_content)
                
                if not cleaned_text.strip():
                    raise WebResponseError("Aucun contenu textuel extrait")
                
                return cleaned_text
                
            except Exception as e:
                raise WebResponseError(f"Erreur extraction: {str(e)}")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de contenu de {url}: {e}")
            raise e
    
    # CAPTURE D'ÉCRAN -----------------------------------------------------------
    
    async def take_screenshot(self, url: str, output_path: Optional[str] = None, full_page: bool = True, enable_images: bool = True) -> bytes:
        """
        Prend une capture d'écran d'une page web (optimisé).
        
        Args:
            url: L'URL de la page à capturer
            output_path: Chemin de sauvegarde (optionnel)
            full_page: Si True, capture la page entière
            enable_images: Si True, charge les images (nécessaire pour screenshot)
        """
        browser = None
        page = None
        playwright = None
        
        try:
            # Pour les screenshots, on a besoin d'un navigateur séparé avec images
            if enable_images:
                playwright = await async_playwright().start()
                browser = await playwright.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-extensions',
                        '--disable-plugins',
                        '--disable-gpu',
                        '--no-first-run'
                    ]
                )
                page = await browser.new_page(
                    viewport={'width': self.viewport_width, 'height': self.viewport_height}
                )
                page.set_default_timeout(self.timeout)
            else:
                browser, page = await self._get_browser()
            
            # Navigation optimisée
            try:
                await page.goto(url, wait_until='networkidle', timeout=self.timeout)
                await page.wait_for_timeout(1000)  # Attente réduite
            except Exception as e:
                raise WebRequestError(f"Navigation impossible vers {url}: {str(e)}")
            
            try:
                # Options de capture optimisées
                screenshot_options = {
                    'full_page': full_page,
                    'type': 'png',
                    'quality': 85 if not full_page else None  # Compression pour vitesse
                }
                
                if output_path:
                    screenshot_options['path'] = output_path
                
                screenshot_data = await page.screenshot(**screenshot_options)
                return screenshot_data
                
            except Exception as e:
                raise WebResponseError(f"Erreur capture: {str(e)}")
                
        finally:
            # Nettoyer uniquement si on a créé un navigateur séparé
            if enable_images and browser:
                await browser.close()
            if enable_images and playwright:
                await playwright.stop()
    
    # PAYLOAD OPENAI ------------------------------------------------------------
    
    async def screenshot_to_openai_payload(self, 
                                          url: str, 
                                          detail: Literal['low', 'high', 'auto'] = 'auto',
                                          full_page: bool = True,
                                          save_to_disk: bool = True) -> dict:
        """Crée un payload OpenAI à partir d'une capture d'écran."""
        try:
            # Nettoyage automatique périodique
            self._cleanup_counter += 1
            if self._cleanup_counter >= self._cleanup_frequency:
                self.cleanup_old_screenshots()
                self._cleanup_counter = 0
            
            # Screenshot optimisé
            screenshot_data = await self.take_screenshot(url, full_page=full_page, enable_images=True)
            
            # Encodage base64
            base64_image = base64.b64encode(screenshot_data).decode('utf-8')
            data_url = f"data:image/png;base64,{base64_image}"
            
            # Sauvegarde optionnelle
            saved_path = None
            if save_to_disk:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                url_safe = re.sub(r'[^\w\-_.]', '_', url.replace('https://', '').replace('http://', ''))[:30]
                filename = f"ss_{url_safe}_{timestamp}.png"
                saved_path = TEMP_DIR / filename
                
                # Écriture rapide
                with open(saved_path, 'wb') as f:
                    f.write(screenshot_data)
            
            return {
                'payload': {
                    'type': 'image_url',
                    'image_url': {
                        'url': data_url,
                        'detail': detail
                    }
                },
                'metadata': {
                    'source_url': url,
                    'timestamp': datetime.now().isoformat(),
                    'size_bytes': len(screenshot_data),
                    'detail_level': detail,
                    'full_page': full_page,
                    'saved_path': str(saved_path) if saved_path else None
                },
                'base64_data': base64_image,
                'data_url': data_url
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du payload OpenAI pour {url}: {e}")
            raise WebResponseError(f"Erreur payload OpenAI: {str(e)}")
    
    # UTILITAIRES ---------------------------------------------------------------
    
    def _clean_text(self, text: str) -> str:
        """
        Nettoie le texte extrait pour le rendre plus lisible par une IA.
        
        Args:
            text: Le texte brut à nettoyer
            
        Returns:
            Le texte nettoyé
        """
        if not text:
            return ""
        
        # Supprimer les espaces multiples et les retours à la ligne excessifs
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Supprimer les caractères de contrôle
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Nettoyer les débuts et fins
        text = text.strip()
        
        # Limiter la longueur si nécessaire (optionnel)
        max_length = 50000  # Environ 12k tokens pour GPT
        if len(text) > max_length:
            text = text[:max_length] + "...\n[CONTENU TRONQUÉ]"
        
        return text

                
async def setup(bot):
    await bot.add_cog(Web(bot))
