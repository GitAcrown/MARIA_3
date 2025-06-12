"""### LLM > Agents
Contient l'implémentation des agents."""

import io
import logging
import os
import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence
from moviepy import VideoFileClip

import discord
import openai
import imageio
from openai import AsyncOpenAI
from pydantic import BaseModel

from .classes import *

logger = logging.getLogger(f'MARIA3.agents')

# CONSTANTS -----------------------------------------------------------------

# Chatbot
CHATBOT_TEMPERATURE = 1.0
CHATBOT_COMPLETION_MODEL = 'gpt-4.1'
CHATBOT_TRANSCRIPTION_MODEL = 'whisper-1'
CHATBOT_MAX_COMPLETION_TOKENS = 450
CHATBOT_CONTEXT_WINDOW = 512 * 20 # ~10k tokens
CHATBOT_MAX_CONTEXT_AGE = timedelta(hours=12)

# Monitor
MONITOR_DEV_PROMPT = """
Tu dois déterminer si MARIA (SELF) doit répondre aux messages fournis marqués d'un '<!>'.

REPONDRE 'YES' QUAND :
- Le message contient une question directe avec "?" semblant adressée à MARIA (par ex. une question subsidiaire)
- Le message contient une demande explicite de réponse (par ex. "Explique-moi", "Peux-tu")
- Le message contient une demande de recherche d'informations (par ex. "Quelle est", "Où se trouve", "Combien")
- Le message est une confirmation à une question oui/non de SELF.

REPONDRE 'NO' QUAND :
- Dans absolument TOUT les cas où y'a un doute sur si il faut répondre ou non.
- Les messages courts sans question comme les remerciements, salutations, réactions à un autre message, etc.
- Les messages qui ne semblent pas être adressés à SELF, ou qui mentionnent un autre utilisateur/bot/application.
- Les messages de "réaction" à la réponse, comme des emojis ou des onomatopées.
- Lorsqu'un message précédent de l'utilisateur n'a pas déjà été répondu par SELF.

FORMAT DE L'HISTORIQUE :
- Les messages sont fournis dans l'ordre chronologique, du plus ancien au plus récent.
- Les messages d'utilisateurs sont sous la forme `[<message.id>] <author.name> (<author.id>) : <message.content>`
- Les autres messages sont précédés de `SELF: ` si c'est un message de l'assistant (MARIA), de `APP/BOT: ` si c'est un message d'une application ou d'un bot tiers.
- Ignorer ce qui se trouve après '-#' dans les messages.

FORMAT :
Réponds en JSON avec : {"actions": [{"message_id": ID, "choice": "YES" ou "NO"}]}
"""
MONITOR_TEMPERATURE = 0.1
MONITOR_COMPLETION_MODEL = 'gpt-4.1-nano'
MONITOR_MAX_HISTORY_WINDOW = 20 # Nombre de messages maximum à récupérer dans l'historique
MONITOR_CONTEXT_RETRIEVING = timedelta(minutes=10) # Age maximum du message le plus ancien à récupérer dans l'historique

# Summary
SUMMARY_DEV_PROMPT = """
Tu dois résumer les messages fournis depuis un salon Discord en un message assez court et concis.

# FORMAT DE L'HISTORIQUE
- Les messages sont fournis dans l'ordre chronologique, du plus ancien au plus récent.
- Les messages d'utilisateurs sont sous la forme `[<message.id>] <author.name> (<author.id>) : <message.content>`
- Les autres messages sont précédés de `SELF: ` si c'est un message de l'assistant (MARIA), de `APP/BOT: ` si c'est un message d'une application ou d'un bot tiers.
- Ignorer ce qui se trouve après '-#' dans les messages.

# RÉPONSE
Tu dois répondre par un résumé court, concis et précis des messages. Quand tu parles d'un utilisateur, tu dois le citer par son pseudonyme seulement.
"""
SUMMARY_TEMPERATURE = 0.1
SUMMARY_COMPLETION_MODEL = 'gpt-4.1-nano'
SUMMARY_MAX_COMPLETION_TOKENS = 512 # 256 tokens
SUMMARY_EVERY_N_TOKENS = 512 * 8 # 4k tokens

# Video analysis
TEMP_DIR = Path('./temp')
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_ANALYSIS_DEV_PROMPT = """A partir des éléments fournis (images et transcription audio) extraits d'une vidéo postée par un utilisateur, réalise une description TRÈS DÉTAILLÉE de la vidéo (personnes, background, actions, textes, etc.). Ne répond qu'avec cette description sans aucun autre texte. Les images sont fournies dans l'ordre chronologique et sont extraites à intervalles égaux."""
VIDEO_ANALYSIS_TEMPERATURE = 0.1
VIDEO_ANALYSIS_COMPLETION_MODEL = 'gpt-4.1-nano'
VIDEO_ANALYSIS_MAX_COMPLETION_TOKENS = CHATBOT_MAX_COMPLETION_TOKENS
VIDEO_ANALYSIS_MAX_ANALYSIS_FILE_SIZE = 20 * 1024 * 1024 # 20 Mo
VIDEO_ANALYSIS_NB_IMAGES_EXTRACTED = 10

# EXCEPTIONS -----------------------------------------------------------------

class GPTWrapperError(Exception):
    pass

class OpenAIError(GPTWrapperError):
    pass

# RESPONSE FORMAT ------------------------------------------------------------

class MessageAction(BaseModel):
    message_id: int
    choice: Literal['YES', 'NO']

class ContextResponse(BaseModel):
    actions: list[MessageAction]

# AGENTS ---------------------------------------------------------------------

class GPTAgent:
    """Classe de base pour les agents GPT."""
    def __init__(self,
                 client: AsyncOpenAI,
                 **kwargs: Any):
        self._client = client
        self.kwargs = kwargs

class ChatbotAgent(GPTAgent):
    """Agent de discussion avec les utilisateurs d'un salon Discord."""
    def __init__(self,
                 client: AsyncOpenAI,
                 developer_prompt_template: Callable[[None], str],
                 temperature: float = CHATBOT_TEMPERATURE,
                 *,
                 completion_model: str = CHATBOT_COMPLETION_MODEL,
                 transcription_model: str = CHATBOT_TRANSCRIPTION_MODEL,
                 max_completion_tokens: int = CHATBOT_MAX_COMPLETION_TOKENS,
                 context_window: int = CHATBOT_CONTEXT_WINDOW,
                 max_context_age: timedelta = CHATBOT_MAX_CONTEXT_AGE,
                 tools: list[Tool] = [],
                 tools_enabled: bool = True,
                 tools_parallel_calls: bool = True,
                 **kwargs: Any):
        super().__init__(client, **kwargs)

        # Paramètres
        self.developer_prompt_template = developer_prompt_template
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.context_window = context_window
        self.max_context_age = max_context_age
        self.tools = tools
        self.tools_enabled = tools_enabled
        self.tools_parallel_calls = tools_parallel_calls

        # Modèles
        self.completion_model = completion_model
        self.transcription_model = transcription_model

        # Contexte
        self._context : Sequence[MessageGroup] = []
        self._last_context_cleanup = datetime.now()

        # Caches
        self.__transcript_cache : dict[str, str] = {}
        self.__video_cache : dict[str, str] = {}

        # Autres arguments
        self.kwargs = kwargs

    # Contexte ----------------------------------------------------
    @property
    def instructions(self) -> DeveloperMessage:
        """Génère le message de développeur."""
        return DeveloperMessage(self.developer_prompt_template())
    
    # Groupes
    
    def get_groups(self, filter: Callable[[MessageGroup], bool] = lambda _: True) -> list[MessageGroup]:  
        """Récupère les groupes de messages qui correspondent au filtre."""
        return [g for g in self._context if filter(g)]
    
    def get_last_group(self, filter: Callable[[MessageGroup], bool] = lambda _: True) -> MessageGroup | None:
        """Récupère le dernier groupe de messages qui correspond au filtre."""
        for g in reversed(self._context):
            if filter(g):
                return g
        return None
    
    def get_group_index(self, group: MessageGroup) -> int:
        """Récupère l'index d'un groupe de messages."""
        return self._context.index(group)
    
    def create_and_insert_group(self, *messages: ContextMessage, **kwargs: Any) -> MessageGroup:
        """Crée un groupe de messages."""
        group = MessageGroup(messages, **kwargs)
        self._context.append(group)
        return group
    
    def insert_group(self, group: MessageGroup, index: int = 0) -> None:
        """Insère un groupe de messages à un index spécifique."""
        self._context.insert(index, group)
    
    def remove_group(self, *groups: MessageGroup) -> None:
        """Supprime un groupe de messages."""
        for group in groups:
            self._context.remove(group)
    
    def flush_history(self) -> None:
        """Réinitialise la mémoire contextuelle de l'assistant."""
        self._context = []

    # Gestion 

    def search_discord_message(self, message: discord.Message) -> MessageGroup | None:
        """Recherche un groupe de messages par message Discord."""
        for group in self._context:
            for m in group.messages:
                if isinstance(m, UserMessage) and m.message == message:
                    return group
        return None
    
    def filter_context(self, filter: Callable[[MessageGroup], bool] = lambda _: True) -> None:
        """Conserve les groupes de messages qui correspondent au filtre et supprime les autres."""
        self._context = [g for g in self._context if filter(g)]
    
    def cleanup_context(self) -> None:
        """Nettoie le contexte."""
        self._context = [g for g in self._context if g.total_token_count < self.context_window]

    def compile_context(self) -> list[ContextMessage]:
        """Compile le contexte pour l'envoi à l'API en optimisant l'utilisation des tokens."""
        self.cleanup_context()
        result = [self.instructions]
        tokens_used = self.instructions.token_count
        token_limit = self.context_window

        # Toujours inclure le dernier groupe complet
        if self._context:
            last_group = self._context[-1]
            last_group_tokens = last_group.total_token_count
            result.extend(last_group.messages)
            tokens_used += last_group_tokens
            token_limit -= tokens_used

        # Ajouter autant de groupes précédents que possible
        for group in reversed(self._context[:-1] if self._context else []):
            group_tokens = group.total_token_count
            if tokens_used + group_tokens > self.context_window:
                break

            for msg in reversed(group.messages):
                result.insert(1, msg)
            tokens_used += group_tokens
        return result
    
    # Outils ----------------------------------------------------

    def get_tools(self, name: str | None = None) -> list[Tool]:
        if name:
            return [t for t in self.tools if t.name == name]
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Tool | None:
        return next((t for t in self.tools if t.name == name), None)
    
    def add_tools(self, *tools: Tool) -> None:
        self.tools.extend(tools)
    
    def remove_tools(self, *tools: Tool) -> None:
        for tool in tools:
            self.tools.remove(tool)

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
    
    # Complétion ----------------------------------------------------
    async def complete_context(self) -> MessageGroup:
        """Exécute une complétion de texte avec le modèle OpenAI et renvoie la réponse.
        
        :return: Groupe de messages contenant la réponse de l'assistant et les éventuelles réponses d'outils
        """
        messages = self.compile_context()

        current_group = self.get_last_group(lambda g: g.awaiting_response)
        if not current_group:
            raise ValueError("Aucun groupe de messages trouvé")
        
        await self.handle_messages_attachments(messages)
        
        payload = [m.payload for m in messages]
        
        try:
            completion = await self._client.chat.completions.create(
                model=self.completion_model,
                messages=payload,
                temperature=self.temperature,
                max_tokens=self.max_completion_tokens,
                tools=[t.to_dict for t in self.get_tools()],
                parallel_tool_calls=True
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
            return await self.complete_context()
        return current_group
    
    async def simple_chat_completion(self, 
                                     messages: list[ContextMessage],
                                     model: str = CHATBOT_COMPLETION_MODEL,
                                     temperature: float = CHATBOT_TEMPERATURE,
                                     max_tokens: int = CHATBOT_MAX_COMPLETION_TOKENS) -> AssistantMessage:
        """Exécute une complétion de texte avec le modèle OpenAI et renvoie la réponse."""
        payload = [m.payload for m in messages]
        completion = await self._client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return AssistantMessage.from_chat_completion(completion)
    
    # Transcription audio ----------------------------------------------------
    async def extract_audio_transcript(self,
                                        audio_file: io.BytesIO | Path | str,
                                        *,
                                        close_binary: bool = True,
                                        unlink_path: bool = True) -> str:
        if isinstance(audio_file, io.BytesIO):
            audio_file.seek(0)
        try:
            transcript = await self._client.audio.transcriptions.create(
                model=self.transcription_model,
                file=audio_file
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

    # Helpers ----------------------------------------------------
    
    def fetch_message_group(self, message: discord.Message) -> MessageGroup | None:
        for group in self._context:
            for m in group.messages:
                if isinstance(m, UserMessage) and m.kwargs.get('message') == message:
                    return group
        return None
    
    def cleanup_caches(self, keep: int = 25) -> None:
        """Nettoie les caches."""
        if len(self.__transcript_cache) > keep:
            keys = list(self.__transcript_cache.keys())
            for key in keys[:-keep]:
                del self.__transcript_cache[key]
                
    # ATTACHMENTS ----------------------------------------------------   

    async def handle_messages_attachments(self, messages: list[ContextMessage]) -> None:
        """Traite les objets attachés aux messages."""
        for message in messages:
            if message.attachments and not message.attachments_processed:
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
                if obj.attachment.url in self.__transcript_cache:
                    transcript = self.__transcript_cache[obj.attachment.url]
                else:
                    try:
                        audio_file = io.BytesIO()
                        audio_file.name = obj.attachment.filename
                        await obj.attachment.save(audio_file, seek_begin=True)
                        transcript = await self.extract_audio_transcript(audio_file=audio_file)
                        self.__transcript_cache[obj.attachment.url] = transcript    
                    except Exception as e:
                        logger.error(f"Erreur de transcription audio : {e}")
                        transcript = f"TRANSCRIPTION_FAILED"
                        if obj.attachment.url in self.__transcript_cache:
                            del self.__transcript_cache[obj.attachment.url]
                    finally:
                        if audio_file:
                            audio_file.close()
                    message.add_components(MetadataTextComponent('AUDIO', filename=obj.attachment.filename, transcript=transcript, url=obj.attachment.url))
                    message.remove_attachments(obj)
            
            # Vidéo -> MetadataTextComponent (transcription audio) + ImageURLComponent (thumbnail)
            if isinstance(obj, VideoAttachment):
                if obj.attachment.filename in self.__video_cache:
                    analysis = self.__video_cache[obj.attachment.filename]
                else:
                    # Vérification de la taille du fichier
                    if obj.attachment.size > VIDEO_ANALYSIS_MAX_ANALYSIS_FILE_SIZE:
                        logger.warning(f"Fichier vidéo trop volumineux : {obj.attachment.filename}")
                        message.add_components(MetadataTextComponent('VIDEO', filename=obj.attachment.filename, size=obj.attachment.size, error='FILE_TOO_LARGE'))
                        message.remove_attachments(obj)
                        continue
                    
                    # Téléchargement du fichier vidéo
                    path = TEMP_DIR / obj.attachment.filename
                    await obj.attachment.save(path, seek_begin=True, use_cached=True)
                    if not path.exists():
                        logger.warning(f"Fichier vidéo introuvable : {path}")
                        message.add_components(MetadataTextComponent('VIDEO', filename=obj.attachment.filename, error='FILE_NOT_FOUND'))
                        message.remove_attachments(obj)
                        continue
                    
                    audio_transcript = ''
                    images = []
                    analysis = ''
                    try:
                        clip = VideoFileClip(str(path))
                        audio = clip.audio
                        if audio:
                            audio_path = path.with_suffix('.wav')
                            audio.write_audiofile(str(audio_path))
                            audio_transcript = await self.extract_audio_transcript(audio_file=audio_path)
                        
                        images = []
                        duration = clip.duration
                        if duration:
                            time_points = [duration * i / VIDEO_ANALYSIS_NB_IMAGES_EXTRACTED for i in range(VIDEO_ANALYSIS_NB_IMAGES_EXTRACTED)] # On extrait des images à intervalles réguliers
                            for t, time_point in enumerate(time_points):
                                frame = clip.get_frame(time_point)
                                frame_path = path.with_stem(f"frame_{t}").with_suffix('.jpg')
                                imageio.imwrite(str(frame_path), frame)
                                images.append(frame_path)
                        
                        analysis = None
                        dev_prompt = DeveloperMessage(VIDEO_ANALYSIS_DEV_PROMPT)
                        ctx_messages : list[ContextMessage] = [dev_prompt]
                        components = []
                        for i, image_path in enumerate(images):
                                with open(image_path, 'rb') as img_file:
                                    encoded = base64.b64encode(img_file.read()).decode('utf-8')
                                    data_url = f"data:image/jpeg;base64,{encoded}"
                                    components.append(ImageURLComponent(data_url, detail='low'))
                        components.append(MetadataTextComponent('VIDEO', filename=obj.attachment.filename, duration=duration, audio_transcript=audio_transcript, images_extracted=len(images)))
                                
                        ctx_messages.append(UserMessage(components))
                
                        analysis = await self.simple_chat_completion(
                            messages=ctx_messages,
                            model=VIDEO_ANALYSIS_COMPLETION_MODEL,
                            temperature=VIDEO_ANALYSIS_TEMPERATURE,
                            max_tokens=VIDEO_ANALYSIS_MAX_COMPLETION_TOKENS
                        )
                        analysis = MetadataTextComponent('VIDEO', filename=obj.attachment.filename, duration=duration, audio_transcript=audio_transcript, images_extracted=len(images), description=analysis.full_text)
                        self.__video_cache[obj.attachment.filename] = analysis
                    except openai.BadRequestError as e:
                        if 'invalid_image_url' in str(e):
                            logger.warning(f"Erreur d'analyse vidéo : {e}")
                            analysis = MetadataTextComponent('VIDEO', filename=obj.attachment.filename, audio_transcript=audio_transcript, images_extracted=len(images), error='INVALID_IMAGE_URL')
                        else:
                            logger.error(f"Erreur d'analyse vidéo : {e}")
                            analysis = MetadataTextComponent('VIDEO', filename=obj.attachment.filename, audio_transcript=audio_transcript, images_extracted=len(images), error='ANALYSIS_FAILED')
                        if obj.attachment.filename in self.__video_cache:
                            del self.__video_cache[obj.attachment.filename]
                    except openai.OpenAIError as e:
                        logger.error(f"Erreur d'analyse vidéo : {e}")
                        analysis = MetadataTextComponent('VIDEO', filename=obj.attachment.filename, audio_transcript=audio_transcript, images_extracted=len(images), error='ANALYSIS_FAILED')
                        if obj.attachment.filename in self.__video_cache:
                            del self.__video_cache[obj.attachment.filename]
                    except FileNotFoundError as e:
                        logger.error(f"Erreur d'analyse vidéo : {e}")
                        analysis = MetadataTextComponent('VIDEO', filename=obj.attachment.filename, audio_transcript=audio_transcript, images_extracted=len(images), error='FILE_NOT_FOUND')
                        if obj.attachment.filename in self.__video_cache:
                            del self.__video_cache[obj.attachment.filename]
                    except Exception as e:
                        logger.error(f"Erreur d'analyse vidéo : {e}")
                        analysis = MetadataTextComponent('VIDEO', filename=obj.attachment.filename, audio_transcript=audio_transcript, images_extracted=len(images), error='ANALYSIS_FAILED')
                        if obj.attachment.filename in self.__video_cache:
                            del self.__video_cache[obj.attachment.filename]
                    finally: # On nettoie tout
                        try:
                            if 'clip' in locals():
                                clip.close()
                            if 'audio' in locals() and audio:
                                audio.close()
                        except Exception as e:
                            logger.error(f"Erreur lors de la fermeture des ressources vidéo : {e}")
                            
                        for image_path in images:
                            if image_path.exists():
                                os.unlink(image_path)
                        if 'audio_path' in locals() and audio_path and audio_path.exists():
                            os.unlink(audio_path)
                        if path.exists():
                            os.unlink(path)
                        
                message.add_components(analysis)
                message.remove_attachments(obj)


class MonitorAgent(GPTAgent):
    """Agent de surveillance des messages d'un salon Discord afin de déterminer si une réponse est nécessaire."""
    def __init__(self,
                 client: AsyncOpenAI,
                 dev_prompt: str = MONITOR_DEV_PROMPT,
                 temperature: float = MONITOR_TEMPERATURE,
                 *,
                 completion_model: str = MONITOR_COMPLETION_MODEL,
                 max_history_window: int = MONITOR_MAX_HISTORY_WINDOW,
                 context_retrieving: timedelta = MONITOR_CONTEXT_RETRIEVING,
                 **kwargs: Any):
        super().__init__(client, **kwargs)

        # Paramètres
        self.dev_prompt = DeveloperMessage(dev_prompt)
        self.temperature = temperature
        self.max_history_window = max_history_window
        self.context_retrieving = context_retrieving

        # Modèles
        self.completion_model = completion_model

    # Contexte ----------------------------------------------------
    def _create_from_discord_history(self, bot_user: discord.User | discord.Member, history: list[discord.Message]) -> MessageGroup:
        """Crée un groupe de messages à partir d'un historique de messages Discord."""
        messages = []
        for message in history:
            if message.author == bot_user:
                messages.append(AssistantMessage(TextComponent(f"SELF: {message.clean_content}")))
            elif message.author.bot:
                messages.append(AssistantMessage(TextComponent(f"APP/BOT: {message.clean_content}")))
            else:
                messages.append(UserMessage.from_discord_message(message, include_attachments=False))
        return MessageGroup(messages)

    # Complétion ----------------------------------------------------
    async def detect_message_reply(self, bot_user: discord.User | discord.Member, message: discord.Message) -> MessageAction:
        """Détecte si le message visé doit être répondu par un assistant."""
        # On reconstitue l'historique
        history = []
        max_age = datetime.now() - self.context_retrieving
        async for msg in message.channel.history(after=max_age):
            history.append(msg)
        # On garde que les [max_history_window] derniers messages
        history = history[-self.max_history_window:]
        # On retire le message visé car il va être marqué séparément
        try:
            history.remove(message)
        except ValueError:
            pass
        context = self._create_from_discord_history(bot_user, history)
        # On insère le dernier message en le marquant pour détecter la réponse
        # Utilisation de la méthode existante avec le format personnalisé
        marked_message = UserMessage.from_discord_message(message, context_format='<!> [{message.id}] {message.author.name} ({message.author.id})', include_attachments=False)
        context.append_messages(marked_message)
        
        # On lance une complétion avec parsing
        payload = [self.dev_prompt.payload] + [m.payload for m in context.messages]
        try:
            completion = await self._client.beta.chat.completions.parse(
                model=self.completion_model,
                messages=payload,
                temperature=self.temperature,
                response_format=ContextResponse
            )
        except openai.BadRequestError as e:
            logger.error(f"Erreur de complétion : {e}")
            raise e
        
        if not completion.choices:
            return MessageAction(message_id=message.id, choice='NO')
        completion_message = completion.choices[0].message
        if completion_message.refusal:
            return MessageAction(message_id=message.id, choice='NO')
        try:
            parsed_response = completion_message.parsed
            # Rechercher l'action pour ce message spécifique
            for action in parsed_response.actions:
                if action.message_id == message.id:
                    return action
            # Si aucune action trouvée pour ce message, retourner NO par défaut
            return MessageAction(message_id=message.id, choice='NO')
        except Exception as e:
            logger.error(f"Erreur de conversion : {e}")
            return MessageAction(message_id=message.id, choice='NO')

class SummaryAgent(GPTAgent):
    """Agent de résumé des messages d'un salon Discord."""
    def __init__(self,
                 client: AsyncOpenAI,
                 dev_prompt: str = SUMMARY_DEV_PROMPT,
                 temperature: float = SUMMARY_TEMPERATURE,
                 *,
                 completion_model: str = SUMMARY_COMPLETION_MODEL,
                 max_completion_tokens: int = SUMMARY_MAX_COMPLETION_TOKENS,
                 every_n_tokens: int = SUMMARY_EVERY_N_TOKENS,
                 **kwargs: Any):
        super().__init__(client, **kwargs)

        # Paramètres
        self.dev_prompt = DeveloperMessage(dev_prompt)
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.every_n_tokens = every_n_tokens

        # Modèles
        self.completion_model = completion_model

        # Historique
        self._history : list[ContextMessage] = []

    class AgentSummary:
        def __init__(self, text: str, history: list[ContextMessage]):
            self.text : str = text
            self.history : list[ContextMessage] = history
            self.start_time : datetime = history[0]._created_at.replace(tzinfo=timezone.utc)
            self.end_time : datetime = history[-1]._created_at.replace(tzinfo=timezone.utc)
            self.authors : list[discord.Member] = [m.message.author for m in history if isinstance(m, UserMessage)]

    # Contexte ----------------------------------------------------

    def add_user_message(self, message: discord.Message) -> None:
        """Ajoute un message d'utilisateur à l'historique."""
        self._history.append(UserMessage.from_discord_message(message, include_attachments=False))
    
    def add_assistant_message(self, message: discord.Message, is_self: bool = False) -> None:
        """Ajoute un message de l'assistant à l'historique."""
        if is_self:
            self._history.append(AssistantMessage(TextComponent(f"SELF: {message.clean_content}")))
        elif message.author.bot:
            self._history.append(AssistantMessage(TextComponent(f"APP/BOT: {message.clean_content}")))
        else:
            self._history.append(AssistantMessage(TextComponent(message.clean_content)))

    def bulk_load_user_messages(self, messages: list[discord.Message]) -> None:
        """Ajoute un ensemble de messages d'utilisateur à l'historique."""
        for message in messages:
            self.add_user_message(message)

    def try_removing_message(self, message: discord.Message) -> bool:
        """Essaye de supprimer un message de l'historique en le retrouvant par son ID."""
        for m in self._history:
            if hasattr(m, 'message') and m.message.id == message.id:
                self._history.remove(m)
                return
        return False
    
    def get_history(self) -> list[ContextMessage]:
        """Récupère l'historique."""
        return self._history
    
    def flush_history(self) -> None:
        """Nettoie l'historique."""
        self._history = []
    
    async def maybe_summarize(self) -> AgentSummary | None:
        """Résume l'historique si le nombre de tokens est supérieur à [every_n_tokens]."""
        total_tokens = sum([m.token_count for m in self._history])
        if total_tokens < self.every_n_tokens:
            return None

        summary = await self.summarize_history()
        self.flush_history()
        return summary

    async def summarize_history(self) -> AgentSummary:
        """Résume l'historique."""
        logger.warning(f"Résumé en cours de l'historique avec {len(self._history)} messages")
        history = self.get_history()
        if not history:
            return None 
        
        # On lance une complétion
        payload = [self.dev_prompt.payload] + [m.payload for m in history]
        try:
            completion = await self._client.chat.completions.create(
                model=self.completion_model,
                messages=payload,
                temperature=self.temperature,
                max_tokens=self.max_completion_tokens
            )
        except openai.BadRequestError as e:
            logger.error(f"Erreur de complétion : {e}")
            raise e
        except openai.OpenAIError as e:
            logger.error(f"Erreur OpenAI : {e}")
            raise e
        except Exception as e:
            logger.error(f"Erreur inconnue : {e}")
            raise e
        
        return self.AgentSummary(completion.choices[0].message.content, history)
    
