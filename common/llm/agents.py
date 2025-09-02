"""### LLM > Agents
Contient l'implémentation des agents."""

import io
import logging
import os
import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence
import zoneinfo
from moviepy import VideoFileClip

import discord
import openai
import imageio
from openai import AsyncOpenAI
from pydantic import BaseModel

from .classes import *

logger = logging.getLogger(f'MARIA3.agents')

# CONSTANTES ------------------------------------------------------

# Fuseau horaire de Paris
PARIS_TZ = zoneinfo.ZoneInfo("Europe/Paris")

# Main Chatbot
MAIN_COMPLETION_MODEL = 'gpt-5-mini'
MAIN_TRANSCRIPTION_MODEL = 'gpt-4o-transcribe'
MAIN_MAX_COMPLETION_TOKENS = 1000
MAIN_CONTEXT_WINDOW = 512 * 32 # 16k tokens
MAIN_CONTEXT_AGE = timedelta(hours=6)

# Opportunistic Chatbot
OPPORTUNISTIC_PROMPT = "Tu as pour objectif d'attribuer un 'score de pertinence de réponse' (compris entre 0 et 100) au message Discord fourni. Ce score doit permettre de mesurer l'utilité pour une IA (nommée MARIA) de répondre au message concerné. Le score doit être élevé si le message est une interrogation de l'utilisateur comme une demande d'informations (internet ou générale), un avis sur quelque chose ou si une réponse est expressément demandée, mais faible si c'est seulement la mention passive du nom de l'IA. Ta réponse est un JSON suivant ce format : {'score': <score de pertinence entre 0 et 100>}."
OPPORTUNISTIC_TEMPERATURE = 0.1
OPPORTUNISTIC_COMPLETION_MODEL = 'gpt-4.1-nano'

class OpportunityScore(BaseModel):
    """Score de pertinence de réponse."""
    score: int

# Analyse vidéo
# Video analysis
TEMP_DIR = Path('./temp')
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_ANALYSIS_DEV_PROMPT = "A partir des éléments fournis (images et transcription audio) qui ont été extraits d'une vidéo, réalise une description EXTREMEMENT DÉTAILLÉE (sujets, actions, scène, apparences etc.). Ne répond qu'avec cette description sans aucun autre texte. Les images sont fournies dans l'ordre chronologique et sont des frames extraites à intervalles égaux de la vidéo."
VIDEO_ANALYSIS_TEMPERATURE = 0.15
VIDEO_ANALYSIS_COMPLETION_MODEL = 'gpt-4.1-mini'
VIDEO_ANALYSIS_AUDIO_MODEL = 'gpt-4o-mini-transcribe'
VIDEO_ANALYSIS_MAX_COMPLETION_TOKENS = 1000
VIDEO_ANALYSIS_MAX_ANALYSIS_FILE_SIZE = 20 * 1024 * 1024 # 20 Mo
VIDEO_ANALYSIS_NB_IMAGES_EXTRACTED = 10

# EXCEPTIONS -----------------------------------------------------------------

class GPTWrapperError(Exception):
    pass

class OpenAIError(GPTWrapperError):
    pass

# AGENTS ------------------------------------------------------

class ChatbotAgent:
    def __init__(self,
                 client: AsyncOpenAI,
                 developer_prompt_template: Callable[[dict], str],
                 developer_prompt_args: dict = None,
                 *,
                 completion_model: str = MAIN_COMPLETION_MODEL,
                 transcription_model: str = MAIN_TRANSCRIPTION_MODEL,
                 max_completion_tokens: int = MAIN_MAX_COMPLETION_TOKENS,
                 context_window: int = MAIN_CONTEXT_WINDOW,
                 context_age: timedelta = MAIN_CONTEXT_AGE,
                 tools: Sequence[Tool] = None,
                 tools_enabled: bool = True,
                 tools_parallel_calls: bool = True,
                 **kwargs):
        
        self.client = client
        
        # Paramètres de l'agent
        self.developer_prompt_template = developer_prompt_template
        self.developer_prompt_args = developer_prompt_args if developer_prompt_args else {}
        self.completion_model = completion_model
        self.transcription_model = transcription_model
        self.max_completion_tokens = max_completion_tokens
        self.context_window = context_window
        self.context_age = context_age
        
        # Outils
        self.tools = list(tools) if tools else []
        self.tools_enabled = tools_enabled
        self.tools_parallel_calls = tools_parallel_calls
        
        # Historique
        self._history_groups: list[MessageGroup] = []
        
        # Caches pour les attachments
        self._transcript_cache: dict[str, str] = {}
        self._video_cache: dict[str, 'MetadataTextComponent'] = {}
        
        # Cache pour les tools compilés (optimisation)
        self._compiled_tools_cache: list[dict] | None = None
    
    # Contrôle de l'historique --------------------------------------
    
    @property
    def developer_message(self) -> DeveloperMessage:
        """Retourne le message du développeur de l'agent."""
        return DeveloperMessage(self.developer_prompt_template(self.developer_prompt_args), created_at=datetime.now(PARIS_TZ))
    
    def get_groups(self, filter: Callable[[MessageGroup], bool] = lambda _: True) -> list[MessageGroup]:
        """Retourne les groupes de messages de l'historique."""
        return [group for group in self._history_groups if filter(group)]
    
    def get_last_group(self, filter: Callable[[MessageGroup], bool] = lambda _: True) -> MessageGroup | None:
        """Retourne le dernier groupe de messages de l'historique."""
        if not self._history_groups:
            return None
        
        # Filtrer et retourner le dernier groupe correspondant
        filtered_groups = [group for group in self._history_groups if filter(group)]
        return filtered_groups[-1] if filtered_groups else None
    
    def create_insert_group(self, *messages: ContextMessage, **shared_kwargs) -> MessageGroup:
        """Crée un groupe de messages à insérer dans l'historique."""
        group = MessageGroup(messages=list(messages), **shared_kwargs)
        self._history_groups.append(group)
        return group
    
    def insert_group(self, group: MessageGroup) -> None:
        """Insère un groupe de messages dans l'historique."""
        if not isinstance(group, MessageGroup):
            raise TypeError("Le groupe doit être une instance de MessageGroup.")
        self._history_groups.append(group)
        
    def remove_group(self, group: MessageGroup) -> None:
        """Supprime un groupe de messages de l'historique."""
        if group in self._history_groups:
            self._history_groups.remove(group)
        else:
            raise ValueError("Le groupe n'est pas dans l'historique.")
        
    def clear_history(self) -> None:
        """Vide l'historique de l'agent."""
        self._history_groups.clear()
        
    # Utilitaires de l'historique ---------------------------------
    
    def corresponding_discord_message(self, discord_msg: discord.Message) -> MessageGroup | None:
        """Retourne le groupe de messages correspondant à un message Discord."""
        for group in self._history_groups:
            for m in group.messages:
                if isinstance(m, UserMessage) and getattr(m, 'message', None) == discord_msg:
                    return group
        return None
    
    def apply_context_filter(self, filter: Callable[[MessageGroup], bool]) -> None:
        """Applique un filtre sur les groupes de messages du contexte."""
        self._history_groups = [group for group in self._history_groups if filter(group)]
    
    # Contexte ---------------------------------
    
    def cleanup_groups(self) -> None:
        """Efface les groupes de messages trop vieux ou dépassant la limite de tokens."""
        # On efface tous les groupes trop vieux
        now = datetime.now(PARIS_TZ)
        self._history_groups = [group for group in self._history_groups if now - group.created_at < self.context_age]
        
        # On efface tous les groupes après le dernier groupe ne respectant plus la limite de tokens
        # Optimisation: Calcul en une seule passe
        total_tokens = 0
        valid_groups = []
        
        for group in reversed(self._history_groups):
            group_tokens = group.total_token_count
            if total_tokens + group_tokens <= self.context_window:
                valid_groups.insert(0, group)
                total_tokens += group_tokens
            else:
                break
                
        self._history_groups = valid_groups
    
    def prepare_context(self) -> list[ContextMessage]:
        """Formate le contexte, décomposant les groupes de messages en une liste de ContextMessage."""
        context = [self.developer_message]
        for group in self._history_groups:
            context.extend(group.messages)
        
        return context
    
    def compile_context(self) -> list[ContextMessage]:
        """Compile le contexte en appliquant le nettoyage et la préparation."""
        self.cleanup_groups()
        return self.prepare_context()
    
    def filter_context(self, filter_func: Callable[[MessageGroup], bool]) -> None:
        """Filtre le contexte selon une fonction de filtrage."""
        self._history_groups = [group for group in self._history_groups if filter_func(group)]
    
    # Outils ---------------------------------
    
    def get_tools(self, name: str | None = None) -> list[Tool]:
        if name:
            return [t for t in self.tools if t.name == name]
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Tool | None:
        return next((t for t in self.tools if t.name == name), None)
    
    def add_tools(self, *tools: Tool) -> None:
        self.tools.extend(tools)
        self._compiled_tools_cache = None  # Invalider le cache
    
    def remove_tools(self, *tools: Tool) -> None:
        for tool in tools:
            self.tools.remove(tool)
        self._compiled_tools_cache = None  # Invalider le cache
    
    def _get_compiled_tools(self) -> list[dict]:
        """Cache les outils compilés pour éviter de les recompiler à chaque appel."""
        if self._compiled_tools_cache is None:
            self._compiled_tools_cache = [t.to_dict for t in self.tools]
        return self._compiled_tools_cache

    async def execute_call_tools(self, tools: list[Tool], tool_calls: list[ToolCall], context: MessageGroup) -> list[ToolResponseMessage]:
        """Exécute les appels d'outils."""
        tool_responses = []
        for tool_call in tool_calls:
            tool = next((t for t in tools if t.name == tool_call.function_name), None)
            if not tool:
                logger.warning(f"Outil '{tool_call.function_name}' non trouvé")
                continue
            
            response = await tool.execute(tool_call, context)
            if response.is_empty:
                logger.warning(f"Outil '{tool_call.function_name}' a retourné une réponse vide")
                continue
            tool_responses.append(response)
        
        return tool_responses
    
    # Complétion ---------------------------------
    
    async def complete_context(self) -> MessageGroup:
        """Exécute une complétion de texte avec le modèle OpenAI et renvoie la réponse.
        
        :return: Groupe de messages contenant la réponse de l'assistant et les éventuelles réponses d'outils
        """
        messages = self.compile_context()
        current_group = self.get_last_group(lambda g: getattr(g, 'awaiting_response', False))
        if not current_group:
            # On récupère le dernier groupe de message contenant un message utilisateur
            current_group = self.get_last_group(lambda g: any(isinstance(m, UserMessage) for m in g.messages))
            if not current_group:
                raise ValueError("Aucun groupe de messages valide trouvé pour la complétion.")
        
        await self.handle_messages_attachments(messages)
        
        payload = [m.payload for m in messages]
        
        # Optimisation: utiliser le cache des outils compilés
        tools_dict = self._get_compiled_tools() if self.tools_enabled else []
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.completion_model,
                messages=payload,
                max_completion_tokens=self.max_completion_tokens,
                reasoning_effort='low',
                verbosity='low',
                tools=tools_dict,
                parallel_tool_calls=self.tools_parallel_calls
            )
        except openai.BadRequestError as e:
            if 'invalid_image_url' in str(e):
                self.filter_context(lambda g: not g.contains_image)
                return await self.complete_context()
            else:
                logger.error(f"Erreur de complétion : {e}")
                raise e
        except openai.OpenAIError as e:
            logger.error(f"Erreur OpenAI : {e}")
            raise e
        except Exception as e:
            logger.error(f"Erreur inconnue : {e}")
            raise e
        
        try:
            amsg = AssistantMessage.from_chat_completion(completion)
        except ValueError as e:
            logger.error(f"Erreur de conversion : {e}")
            raise e
        
        current_group.append_messages(amsg)
        if amsg.tool_calls:
            tool_responses = await self.execute_call_tools(self.get_tools(), amsg.tool_calls, current_group)
            current_group.append_messages(*tool_responses)
            # Éviter la récursion infinie en vérifiant que les tools ont bien répondu
            if tool_responses:
                return await self.complete_context()
            else:
                logger.warning("Aucune réponse d'outil valide, arrêt de la récursion")
        return current_group
    
    # Transcription audio ----------------------------------------------------
    async def extract_audio_transcript(self,
                                        audio_file: io.BytesIO | Path | str,
                                        *,
                                        model: str | None = None,
                                        prompt: str = '',
                                        close_binary: bool = True,
                                        unlink_path: bool = True) -> str:
        if isinstance(audio_file, io.BytesIO):
            audio_file.seek(0)
        
        try:
            transcript = await self.client.audio.transcriptions.create(
                model=model if model else self.transcription_model,
                file=audio_file,
                prompt=prompt
            )
        except openai.BadRequestError as e:
            logger.error(f"Erreur de transcription : {e}")
            raise e
        except openai.OpenAIError as e:
            logger.error(f"Erreur OpenAI : {e}")
            raise e
        except Exception as e:
            logger.error(f"Erreur inconnue : {e}")
            raise e
        
        if isinstance(audio_file, io.BytesIO) and close_binary:
            audio_file.close()
        if isinstance(audio_file, (str, Path)) and unlink_path:
            try:
                os.unlink(audio_file)
            except OSError:
                pass
        return transcript.text
    
    # Chat completions simples ------------------------------------

    async def simple_chat_completion(self, 
                                   messages: list[ContextMessage], 
                                   model: str, 
                                   temperature: float, 
                                   max_tokens: int) -> 'AssistantMessage':
        """Effectue une complétion simple sans outils."""
        payload = [m.payload for m in messages]
        
        completion = await self.client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return AssistantMessage.from_chat_completion(completion)
    
    # Helpers ----------------------------------------------------
    
    def fetch_message_group(self, message: discord.Message) -> MessageGroup | None:
        """Trouve le groupe de messages correspondant à un message Discord."""
        for group in self._history_groups:
            for m in group.messages:
                if isinstance(m, UserMessage) and getattr(m, 'message', None) == message:
                    return group
        return None
    
    def cleanup_caches(self, keep: int = 25) -> None:
        """Nettoie les caches pour éviter une consommation mémoire excessive."""
        if len(self._transcript_cache) > keep:
            # Garder seulement les 'keep' derniers éléments
            items = list(self._transcript_cache.items())
            self._transcript_cache = dict(items[-keep:])
            
        if len(self._video_cache) > keep:
            items = list(self._video_cache.items())
            self._video_cache = dict(items[-keep:])
    
    def get_cache_stats(self) -> dict:
        """Retourne les statistiques des caches pour monitoring."""
        return {
            'transcript_cache_size': len(self._transcript_cache),
            'video_cache_size': len(self._video_cache),
            'tools_cache_valid': self._compiled_tools_cache is not None,
            'history_groups_count': len(self._history_groups)
        }
    
    # ATTACHMENTS ----------------------------------------------------   

    async def handle_messages_attachments(self, messages: list[ContextMessage]) -> None:
        """Traite les objets attachés aux messages."""
        for message in messages:
            if (hasattr(message, 'attachments') and 
                message.attachments and 
                not getattr(message, 'attachments_processed', False)):
                await self.process_attachments(message)
    
    # Traitement des objets
    async def process_attachments(self, message: ContextMessage) -> None:
        """Traite les objets attachés au message et les convertit en composants de contenu."""
        if not message.attachments:
            return
        self.cleanup_caches()
        
        for obj in message.attachments:
            # Message audio -> MetadataTextComponent
            if isinstance(obj, AudioAttachment):
                await self._process_audio_attachment(obj, message)
            
            # Vidéo -> MetadataTextComponent (transcription audio) + ImageURLComponent (thumbnail)
            elif isinstance(obj, VideoAttachment):
                await self._process_video_attachment(obj, message)
            
            # Fichiers texte -> TextComponent
            elif isinstance(obj, TextFileAttachment):
                await self._process_text_file_attachment(obj, message)

    async def _process_audio_attachment(self, obj: 'AudioAttachment', message: 'ContextMessage') -> bool:
        """Traite un attachment audio et retourne True si traité avec succès."""
        if obj.attachment.url in self._transcript_cache:
            transcript = self._transcript_cache[obj.attachment.url]
        else:
            audio_file = None
            try:
                audio_file = io.BytesIO()
                audio_file.name = obj.attachment.filename
                await obj.attachment.save(audio_file, seek_begin=True)
                transcript = await self.extract_audio_transcript(audio_file=audio_file)
                self._transcript_cache[obj.attachment.url] = transcript
            except Exception as e:
                logger.error(f"Erreur de transcription audio : {e}")
                transcript = "TRANSCRIPTION_FAILED"
                # Supprimer de cache en cas d'erreur
                self._transcript_cache.pop(obj.attachment.url, None)
            finally:
                if audio_file:
                    audio_file.close()
        
        message.add_components(MetadataTextComponent('AUDIO', 
                                                   filename=obj.attachment.filename, 
                                                   transcript=transcript, 
                                                   url=obj.attachment.url))
        message.remove_attachments(obj)
        return True
    
    async def _process_video_attachment(self, obj: 'VideoAttachment', message: 'ContextMessage') -> bool:
        """Traite un attachment vidéo et retourne True si traité avec succès."""
        if obj.attachment.filename in self._video_cache:
            analysis = self._video_cache[obj.attachment.filename]
            message.add_components(analysis)
            message.remove_attachments(obj)
            return True
        
        # Vérification de la taille du fichier
        if obj.attachment.size > VIDEO_ANALYSIS_MAX_ANALYSIS_FILE_SIZE:
            logger.warning(f"Fichier vidéo trop volumineux : {obj.attachment.filename}")
            message.add_components(MetadataTextComponent('VIDEO', 
                                                       filename=obj.attachment.filename, 
                                                       size=obj.attachment.size, 
                                                       error='FILE_TOO_LARGE'))
            message.remove_attachments(obj)
            return False
        
        # Téléchargement et traitement du fichier vidéo
        path = TEMP_DIR / obj.attachment.filename
        await obj.attachment.save(path, seek_begin=True, use_cached=True)
        
        if not path.exists():
            logger.warning(f"Fichier vidéo introuvable : {path}")
            message.add_components(MetadataTextComponent('VIDEO', 
                                                       filename=obj.attachment.filename, 
                                                       error='FILE_NOT_FOUND'))
            message.remove_attachments(obj)
            return False
        
        analysis = await self._analyze_video_file(path, obj.attachment.filename)
        
        message.add_components(analysis)
        message.remove_attachments(obj)
        return True
    
    async def _analyze_video_file(self, path: Path, filename: str) -> 'MetadataTextComponent':
        """Analyse un fichier vidéo et retourne les métadonnées."""
        audio_transcript = ''
        images = []
        duration = 0
        
        clip = None
        audio = None
        audio_path = None
        
        try:
            clip = VideoFileClip(str(path))
            duration = getattr(clip, 'duration', 0) or 0
            audio = getattr(clip, 'audio', None)
            
            # Extraction audio
            if audio:
                audio_path = path.with_suffix('.wav')
                try:
                    # Essayer avec les nouveaux paramètres
                    audio.write_audiofile(str(audio_path), verbose=False, logger=None)
                    audio_transcript = await self.extract_audio_transcript(audio_file=audio_path, model=VIDEO_ANALYSIS_AUDIO_MODEL)
                except TypeError as te:
                    # Fallback pour les anciennes versions de moviepy
                    try:
                        audio.write_audiofile(str(audio_path))
                        audio_transcript = await self.extract_audio_transcript(audio_file=audio_path, model=VIDEO_ANALYSIS_AUDIO_MODEL)
                    except Exception as e:
                        logger.warning(f"Erreur extraction audio (fallback) : {e}")
                        audio_transcript = "AUDIO_EXTRACTION_FAILED"
                except Exception as e:
                    logger.warning(f"Erreur extraction audio : {e}")
                    audio_transcript = "AUDIO_EXTRACTION_FAILED"
            
            # Extraction d'images
            if duration and duration > 0:
                time_points = [duration * i / VIDEO_ANALYSIS_NB_IMAGES_EXTRACTED 
                             for i in range(VIDEO_ANALYSIS_NB_IMAGES_EXTRACTED)]
                
                for t, time_point in enumerate(time_points):
                    try:
                        frame = clip.get_frame(time_point)
                        frame_path = path.with_stem(f"frame_{t}").with_suffix('.jpg')
                        imageio.imwrite(str(frame_path), frame)
                        images.append(frame_path)
                    except Exception as e:
                        logger.warning(f"Erreur extraction frame {t} : {e}")
                        continue
            
            # Analyse par IA (seulement si on a des images)
            if images:
                description = await self._analyze_video_content(images, filename, duration, audio_transcript)
            else:
                description = "NO_FRAMES_EXTRACTED"
            
            analysis = MetadataTextComponent('VIDEO', 
                                           filename=filename, 
                                           duration=duration, 
                                           audio_transcript=audio_transcript, 
                                           images_extracted=len(images), 
                                           description=description)
            
            # Mise en cache
            self._video_cache[filename] = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"Erreur d'analyse vidéo : {e}")
            error_type = self._get_video_error_type(e)
            
            analysis = MetadataTextComponent('VIDEO', 
                                           filename=filename, 
                                           audio_transcript=audio_transcript, 
                                           images_extracted=len(images), 
                                           error=error_type)
            
            # Supprimer du cache en cas d'erreur
            self._video_cache.pop(filename, None)
            return analysis
            
        finally:
            # Nettoyage des ressources
            await self._cleanup_video_resources(clip, audio, images, audio_path, path)
    
    async def _analyze_video_content(self, images: list[Path], filename: str, duration: float, audio_transcript: str) -> str:
        """Analyse le contenu vidéo avec l'IA."""
        try:
            dev_prompt = DeveloperMessage(VIDEO_ANALYSIS_DEV_PROMPT)
            ctx_messages = [dev_prompt]
            components = []
            
            # Ajouter les images (avec vérification d'existence)
            for image_path in images:
                if not image_path.exists():
                    logger.warning(f"Image frame introuvable : {image_path}")
                    continue
                    
                try:
                    with open(image_path, 'rb') as img_file:
                        encoded = base64.b64encode(img_file.read()).decode('utf-8')
                        data_url = f"data:image/jpeg;base64,{encoded}"
                        components.append(ImageURLComponent(data_url, detail='low'))
                except Exception as e:
                    logger.warning(f"Erreur lors de l'encodage de l'image {image_path}: {e}")
                    continue
            
            # Vérifier qu'on a au moins une image
            if not components:
                logger.warning("Aucune image valide pour l'analyse vidéo")
                return "NO_VALID_IMAGES"
            
            # Ajouter les métadonnées
            components.append(MetadataTextComponent('VIDEO', 
                                                  filename=filename, 
                                                  duration=duration, 
                                                  audio_transcript=audio_transcript, 
                                                  images_extracted=len(images)))
            
            ctx_messages.append(UserMessage(components))
            
            result = await self.simple_chat_completion(
                messages=ctx_messages,
                model=VIDEO_ANALYSIS_COMPLETION_MODEL,
                temperature=VIDEO_ANALYSIS_TEMPERATURE,
                max_tokens=VIDEO_ANALYSIS_MAX_COMPLETION_TOKENS
            )
            
            return getattr(result, 'full_text', str(result))
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse IA de la vidéo : {e}")
            return f"ANALYSIS_FAILED: {str(e)}"
    
    def _get_video_error_type(self, error: Exception) -> str:
        """Détermine le type d'erreur vidéo."""
        if isinstance(error, openai.BadRequestError):
            if 'invalid_image_url' in str(error):
                return 'INVALID_IMAGE_URL'
            return 'ANALYSIS_FAILED'
        elif isinstance(error, openai.OpenAIError):
            return 'ANALYSIS_FAILED'
        elif isinstance(error, FileNotFoundError):
            return 'FILE_NOT_FOUND'
        else:
            return 'ANALYSIS_FAILED'
    
    async def _cleanup_video_resources(self, clip, audio, images: list[Path], audio_path: Path | None, video_path: Path):
        """Nettoie toutes les ressources vidéo."""
        try:
            # Fermer les ressources multimedia
            if clip:
                clip.close()
            if audio:
                audio.close()
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture des ressources vidéo : {e}")
        
        # Supprimer les fichiers temporaires
        for image_path in images:
            try:
                if image_path.exists():
                    os.unlink(image_path)
            except OSError:
                pass
        
        if audio_path and audio_path.exists():
            try:
                os.unlink(audio_path)
            except OSError:
                pass
        
        if video_path.exists():
            try:
                os.unlink(video_path)
            except OSError:
                pass

    async def _process_text_file_attachment(self, obj: 'TextFileAttachment', message: 'ContextMessage') -> bool:
        """Traite un attachment de fichier texte et retourne True si traité avec succès."""
        try:
            # Vérification de la taille du fichier (limite à 1MB pour les fichiers texte)
            max_size = 1024 * 1024  # 1MB
            if obj.attachment.size > max_size:
                logger.warning(f"Fichier texte trop volumineux : {obj.attachment.filename} ({obj.attachment.size} bytes)")
                message.add_components(MetadataTextComponent('TEXT_FILE', 
                                                           filename=obj.attachment.filename, 
                                                           size=obj.attachment.size, 
                                                           error='FILE_TOO_LARGE'))
                message.remove_attachments(obj)
                return False
            
            # Téléchargement du fichier
            content_bytes = await obj.attachment.read()
            
            # Tentative de décodage du texte avec différents encodages
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
            content = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    content = content_bytes.decode(encoding)
                    used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                logger.warning(f"Impossible de décoder le fichier texte : {obj.attachment.filename}")
                message.add_components(MetadataTextComponent('TEXT_FILE', 
                                                           filename=obj.attachment.filename, 
                                                           error='ENCODING_ERROR'))
                message.remove_attachments(obj)
                return False
            
            # Limitation de la longueur du contenu (100k caractères max)
            max_content_length = 100000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "\n... [CONTENU TRONQUÉ]"
                truncated = True
            else:
                truncated = False
            
            # Création du composant avec métadonnées et contenu
            file_extension = obj.attachment.filename.split('.')[-1].lower() if '.' in obj.attachment.filename else 'txt'
            
            # Ajout des métadonnées sur le fichier
            message.add_components(MetadataTextComponent('TEXT_FILE', 
                                                       filename=obj.attachment.filename,
                                                       size=obj.attachment.size,
                                                       encoding=used_encoding,
                                                       extension=file_extension,
                                                       truncated=truncated))
            
            # Ajout du contenu du fichier
            formatted_content = f"```{file_extension}\n{content}\n```"
            message.add_components(TextComponent(formatted_content))
            
            message.remove_attachments(obj)
            logger.info(f"Fichier texte traité avec succès : {obj.attachment.filename}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du fichier texte {obj.attachment.filename} : {e}")
            message.add_components(MetadataTextComponent('TEXT_FILE', 
                                                       filename=obj.attachment.filename, 
                                                       error='PROCESSING_ERROR'))
            message.remove_attachments(obj)
            return False

class OpportunisticAgent:
    """Agent détectant l'opportunité de répondre à un message en attribuant un score de pertinence."""
    def __init__(self,
                 client: AsyncOpenAI,
                 developer_prompt: str = OPPORTUNISTIC_PROMPT,
                 temperature: float = OPPORTUNISTIC_TEMPERATURE,
                 completion_model: str = OPPORTUNISTIC_COMPLETION_MODEL):
        self.client = client
        self.developer_prompt = developer_prompt
        self.temperature = temperature
        self.completion_model = completion_model
    
    async def score_message(self, message: discord.Message) -> int:
        """Attribue un score de pertinence à un message Discord."""
        if not message.clean_content:
            return 0
        
        payload = [
            {
                "role": "system",
                "content": self.developer_prompt
            },
            {
                "role": "user",
                "content": message.clean_content
            }
        ]
        
        try:
            completion = await self.client.beta.chat.completions.parse(
                model=self.completion_model,
                messages=payload,
                temperature=self.temperature,
                response_format=OpportunityScore
            )
        except openai.OpenAIError as e:
            logger.error(f"Erreur OpenAI lors de l'attribution du score : {e}")
            raise e
        except Exception as e:
            logger.error(f"Erreur inconnue lors de l'attribution du score : {e}")
            raise e
        
        response = completion.choices[0].message
        if response.refusal:
            logger.warning(f"Réponse refusée pour le message {message.id}")
            return 0
        try:
            score = response.parsed.score
            
            if score < 0:
                score = 0
            elif score > 100:
                score = 100
            return score
        except Exception as e:
            logger.error(f"Erreur de parsing du score pour le message {message.id} : {e}")
            return 0
        
    
        