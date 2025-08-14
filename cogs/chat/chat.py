import io
import logging
import os
import random
import re
import zoneinfo
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Union
import numexpr as ne

import discord
from discord import Interaction, app_commands
from discord.ext import commands, tasks
from moviepy import VideoFileClip
from openai import AsyncOpenAI
from pydantic import BaseModel

from common import dataio
from common.llm.agents import *
from common.llm.classes import *

logger = logging.getLogger(f'MARIA3.{__name__.split(".")[-1]}')

# CONSTANTES ----------------------------------------------------

# Fuseau horaire de Paris
PARIS_TZ = zoneinfo.ZoneInfo("Europe/Paris")

DEVELOPER_PROMPT_TEMPLATE = lambda args: f"""Tu es MARIA, assistante IA conversant sur un salon écrit Discord.
[STYLE]
Être concise, directe et informelle.
Adapter le ton à celui des interlocuteurs, dans un contexte de discussion en ligne.
Ne pas proposer de follow-up après une réponse. 
[HISTORIQUE DE CONVERSATION]
Les messages du salon sont fournis dans le format '[message_id] user_name (user_id) : content'. Ne pas formatter tes messages de cette manière.
Les données de pièce jointes sont fournies entre '<>'.
[META]
Date actuelle: {args['weekday']} {args['datetime']} (Heure de Paris)
Limite de connaissance: Juin 2024
[OUTILS]
INFOS UTILISATEUR: Consulter et mettre à jour les informations personnelles de l'utilisateur
RECHERCHE WEB: Rechercher les données récentes sur un sujet et extraire des données des pages web
CALCULS MATHÉMATIQUES: Évaluer des expressions mathématiques et convertir des unités
[CONSIGNES]
Utiliser et combiner les outils de manière proactive et sans modération.
Jamais inventer d'informations. Chercher sur internet si tu ne sais pas ou que les faits sont récents. Toujours vérifier les affirmations des utilisateurs.
Lorsque tu veux modifier les informations de l'utilisateur, consulte d'abord les actuelles pour ne pas écraser des données importantes. Le demandeur ne peut modifier que ses propres informations et doit formuler la demande de manière EXPLICITE.
Utiliser le markdown Discord lorsque pertinent. Mettre les liens entre <> et les données de tableaux ou le code entre ```.
"""

# PARAMETRES -----------------------------------------------------

STATUS_UPDATE_INTERVAL = 120  # Intervalle de mise à jour du statut en minutes
VALID_CHATBOT_CHANNELS = Union[discord.TextChannel, discord.VoiceChannel, discord.Thread]
MAX_EDITION_AGE = timedelta(minutes=2)  # Age maximal des messages pour l'édition

# Répertoire temporaire
TEMP_DIR = Path('./temp')
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

# UI ========================================================================

class UserInfoModal(discord.ui.Modal, title="Mémoire de MARIA"):
    def __init__(self) -> None:
        super().__init__(timeout=None)
        self.userinfo = discord.ui.TextInput(
            label="Informations personnelles",
            style=discord.TextStyle.long,
            placeholder="Vos informations à partager avec MARIA",
            min_length=0,
            max_length=500
        )
        self.add_item(self.userinfo)
        
    async def on_submit(self, interaction: Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        return self.stop()
        
    async def on_error(self, interaction: Interaction, error: Exception) -> None:
        return await interaction.response.send_message(f"**Erreur** × {error}", ephemeral=True)

class TranscriptPrompt(discord.ui.Modal, title="Indications de transcription"):
    def __init__(self) -> None:
        super().__init__(timeout=None)
        self.audioprompt = discord.ui.TextInput(
            label="Prompt de transcription",
            style=discord.TextStyle.short,
            placeholder="Indications pour la transcription audio",
            min_length=0,
            max_length=200
        )
        self.add_item(self.audioprompt)
        
    async def on_submit(self, interaction: Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        return self.stop()
        
    async def on_error(self, interaction: Interaction, error: Exception) -> None:
        return await interaction.response.send_message(f"**Erreur** × {error}", ephemeral=True)

# CLASSES -----------------------------------------------------

class StatusUpdaterAgent:
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.model = "gpt-4.1-nano"
        self.system_prompt = """
        Tu dois créer un très court (3-4 mots MAXIMUM) TEXTE DE STATUT DISCORD pour un chatbot IA (appelée MARIA) qui se genre au féminin. 
        
        CONSIGNES:
        - Le statut doit être grammaticalement correct et avoir du sens, même s'il est court
        - Privilégier des phrases courtes mais complètes ou des expressions idiomatiques connues
        - Utilise l'humour, l'autodérision et le sarcasme avec des formulations correctes
        - Refs à la pop culture encouragés (memes internet récents et francophones, jeux vidéo, séries, films, anime etc.)
        - Le statut est réalisé du point de vue de l'IA, mais éviter les phrases descriptives comme "je suis ..."
        - Pas de jargon technique, langage d'IA ou de termes clichés liés à la technologie
        - Préférer le français, anglais seulement si références culturelles pertinentes
        - Pas d'emoji, pas de point en fin de phrase
        - Exemples d'inspiration (ne pas copier) : "Bug en cours", "Café requis", "Mode sieste", "Erreur 404", "Meilleure que HAL", "R2D2 sur nous"
        
        La réponse doit être un JSON avec la clé "status" contenant le texte du statut, sans autres informations.
        """

    class MessageStatus(BaseModel):
        status: str

    async def get_status(self) -> str:
        """Récupère un statut en fonction de l'humeur."""
        prompt = self.system_prompt
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "developer", "content": prompt}],
            temperature=1.1,
            max_completion_tokens=50,
            response_format=self.MessageStatus
        )
        if not response.choices[0].message.parsed:
            raise Exception("Erreur lors de la récupération du statut.")
        return response.choices[0].message.parsed.status

class ChannelChatSession:
    def __init__(self, 
                 cog: 'Chat',
                 agent: ChatbotAgent,
                 *,
                 answer_modes: Literal['off', 'strict', 'opportunistic', 'greedy'] = 'opportunistic'):
        self.cog = cog
        self.agent = agent
        
        # Paramètres de session
        self.answer_modes = answer_modes
        
    # Complétion de texte
    async def append_message(self, message: discord.Message) -> MessageGroup:
        """Ajoute un message à la session et retourne le groupe de messages."""
        if message.author.bot:
            return 
        
        if message.reference and message.reference.resolved:
            ref_message = UserMessage.from_discord_message(message.reference.resolved)
            ctx_message = UserMessage.from_discord_message(message)
            ctx_message.add_components(MetadataTextComponent('REFERENCE', message_id=message.reference.resolved.id))
            return self.agent.create_insert_group(ref_message, ctx_message)
        else:
            ctx_message = UserMessage.from_discord_message(message)
            return self.agent.create_insert_group(ctx_message)
        
    async def get_answer(self) -> str:
        """Récupère la réponse à un message."""
        group = await self.agent.complete_context()
        return group.last_completion.full_text
    
    # Transcription audio
    async def extract_audio(self, message: discord.Message) -> io.BytesIO | Path | None:
        """Extrait le fichier audio d'un message (audio ou vidéo)."""
        for attachment in message.attachments:
            # Message audio
            if attachment.content_type and attachment.content_type.startswith('audio'):
                buffer = io.BytesIO()
                buffer.name = attachment.filename
                await attachment.save(buffer, seek_begin=True)
                return buffer
            # Vidéo
            elif attachment.content_type and attachment.content_type.startswith('video'):
                path = TEMP_DIR / f"{attachment.filename}_{random.randint(1000, 9999)}"
                path = path.with_suffix('.mp4')
                await attachment.save(path)
                clip = VideoFileClip(str(path))
                audio = clip.audio
                if not audio:
                    return None
                audio_path = path.with_suffix('.wav')
                try:
                    # Essayer avec les nouveaux paramètres
                    audio.write_audiofile(str(audio_path), verbose=False, logger=None)
                except TypeError:
                    # Fallback pour les anciennes versions de moviepy
                    audio.write_audiofile(str(audio_path))
                clip.close()
                
                os.remove(str(path))
                return audio_path
        return None
    
    async def get_audio_transcript(self, 
                                   file: io.BytesIO | Path,
                                   *,
                                   prompt: str = '',
                                   return_type: Literal['text', 'component'] = 'text') -> str | MetadataTextComponent:
        """Récupère le texte d'un message audio."""
        transcript = await self.agent.extract_audio_transcript(file, prompt=prompt)
        if return_type == 'text':
            return transcript
        elif return_type == 'component':
            return MetadataTextComponent('AUDIO', filename=file.name, transcript=transcript)

class Chat(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = dataio.get_instance(self)
        
        # Paramètres du chatbot sur les serveurs
        guild_config = dataio.DictTableBuilder(
            name='guild_config',
            default_values={
                'chatbot_mode': 'opportunistic',
                'opportunist_threshold': 50  # Seuil de score pour le mode 'opportunistic'
            }
        )
        self.data.map_builders(discord.Guild, guild_config)
        
        # Informations personnalisées des utilisateurs
        user_custom = dataio.TableBuilder(
            '''CREATE TABLE IF NOT EXISTS user_custom (
                user_id INTEGER PRIMARY KEY,
                infos TEXT,
                last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )'''
        )

        self.data.map_builders('global', user_custom)
        
        # Agents
        self._gptclient = AsyncOpenAI(
            api_key=self.bot.config['OPENAI_API_KEY']
        )
        self._opportunistic_agent = OpportunisticAgent(client=self._gptclient)
        self._status_updater_agent = StatusUpdaterAgent(self._gptclient)
        
        self.update_status.start()
        
        # Menu contextuel
        self.ctx_audio_transcript = app_commands.ContextMenu(
            name="Transcription audio guidée",
            callback=self.transcript_audio_callback)
        self.bot.tree.add_command(self.ctx_audio_transcript)
        
        # Sessions de chat
        self._SESSIONS : dict[int, ChannelChatSession] = {}
        
        # Messages déjà traités - utilise une deque pour maintenir l'ordre et limiter la taille
        self._processed_messages = deque(maxlen=100)  # Pour éviter les doublons
        
        # Outils
        self.GLOBAL_TOOLS = [
            Tool(
                name='get_user_info',
                description="Consulter les informations personnelles de l'utilisateur.",
                properties={
                    'user_id': {
                        'type': 'integer',
                        'description': "L'ID de l'utilisateur"
                    }
                },
                function=self._tool_get_user_info
            ),
            Tool(
                name='update_user_info',
                description="Mettre à jour les informations personnelles de l'utilisateur. Ne JAMAIS changer avant d'avoir préalablement consulté les infos de l'utilisateur avec 'get_user_info'.",
                properties={
                    'user_id': {
                        'type': 'integer',
                        'description': "L'ID de l'utilisateur"
                    },
                    'infos': {
                        'type': 'string',
                        'description': "Les nouvelles informations à enregistrer pour l'utilisateur (écrase les précédentes, max. 500 caractères)"
                    }
                },
                function=self._tool_update_user_info
            ),
            Tool(
                name='math_eval',
                description='Évalue des expressions mathématiques. Utilise la syntaxe Python standard.',
                properties={
                    'expression': {
                        'type': 'string',
                        'description': "L'expression mathématique à évaluer"
                    }
                },
                function=self._tool_math_eval
            )
        ]
        
    async def cog_unload(self):
        # Arrêt de la tâche de mise à jour du statut
        await self.update_status.stop()
        # Fermeture des clients
        await self._gptclient.close()
        await self._opportunistic_agent.client.close()
        await self._status_updater_agent.client.close()
        # Fermeture de la base de données
        self.data.close_all()
        
    # Loop ====================================================================

    # Mise à jour du statut à intervalle régulier
    @tasks.loop(minutes=STATUS_UPDATE_INTERVAL)
    async def update_status(self):
        new_status = await self._status_updater_agent.get_status()
        await self.bot.change_presence(activity=discord.Activity(type=discord.ActivityType.custom, name='custom', state=new_status))
        logger.info(f"i --- Statut mis à jour : {new_status}")
        
    @update_status.before_loop
    async def before_update_status(self):
        await self.bot.wait_until_ready()  
        
    # DB -------------------------------------------------
    
    # Paramètres de serveur
    def get_guild_config(self, guild: discord.Guild, key: str, cast: Union[type, None] = None) -> Union[str, int, bool, None]:
        """Récupère une clé de configuration du serveur."""
        value = self.data.get(guild).get_dict_value('guild_config', key)
        if cast is not None and value is not None:
            return cast(value)
        return value
    
    def get_full_guild_config(self, guild: discord.Guild) -> dict[str, Union[str, int, bool]]:
        """Récupère la configuration complète du serveur."""
        return self.data.get(guild).get_dict_values('guild_config')
    
    def set_guild_config(self, guild: discord.Guild, key: str, value: Union[str, int, bool]) -> None:
        """Met à jour une clé de configuration du serveur."""
        self.data.get(guild).set_dict_value('guild_config', key, value)
        
    # Infos utilisateur
    def get_user_custom(self, user: discord.User | discord.Member) -> str | None:
        """Récupère les informations personnalisées d'un utilisateur."""
        result = self.data.get('global').fetchone('SELECT infos FROM user_custom WHERE user_id = ?', user.id)
        return result['infos'] if result else None
    
    def update_user_custom(self, user: discord.User | discord.Member, infos: str) -> None:
        """Met à jour les informations personnalisées d'un utilisateur."""
        self.data.get('global').execute(
            'INSERT OR REPLACE INTO user_custom (user_id, infos) VALUES (?, ?)',
            user.id, infos
        )
        
    def remove_user_custom(self, user: discord.User | discord.Member) -> None:
        """Supprime les informations personnalisées d'un utilisateur."""
        self.data.get('global').execute('DELETE FROM user_custom WHERE user_id = ?', user.id)
        
    # Chatbots --------------------------------------------
    
    async def get_channel_chat_session(self, channel: VALID_CHATBOT_CHANNELS) -> ChannelChatSession:
        """Récupère ou crée une session de chat pour un canal."""
        if channel.id not in self._SESSIONS:
            # Crée un agent de chat
            dev_prompt = DEVELOPER_PROMPT_TEMPLATE({
                'weekday': datetime.now(PARIS_TZ).strftime('%A'),
                'datetime': datetime.now(PARIS_TZ).strftime('%Y-%m-%d %H:%M:%S')
            })
            self.populate_tools()
            
            agent = ChatbotAgent(
                client=self._gptclient,
                developer_prompt=dev_prompt,
                tools=self.GLOBAL_TOOLS
            )
        
            self._SESSIONS[channel.id] = ChannelChatSession(
                cog=self,
                agent=agent,
                answer_modes=self.get_guild_config(channel.guild, 'chatbot_mode', str)
            )
            
        return self._SESSIONS[channel.id]
    
    async def remove_channel_chat_session(self, channel: VALID_CHATBOT_CHANNELS) -> None:
        """Supprime la session de chat pour un canal."""
        if channel in self._SESSIONS:
            del self._SESSIONS[channel]
            
    # Outils ------------------------------------------------
    
    def populate_tools(self):
        for cog in self.bot.cogs.values():
            if cog.qualified_name == self.qualified_name:
                continue
            if hasattr(cog, 'GLOBAL_TOOLS'):
                for tool in cog.GLOBAL_TOOLS:
                    if tool.name not in (t.name for t in self.GLOBAL_TOOLS):
                        self.GLOBAL_TOOLS.append(tool)
                        logger.info(f"i --- Outil '{tool.name}' ajouté depuis '{cog.qualified_name}'")
                        
    def _tool_get_user_info(self, tool_call: ToolCall, context: MessageGroup) -> ToolResponseMessage:
        """Outil pour récupérer les informations utilisateur."""
        user_id = tool_call.arguments.get('user_id')
        if user_id is None:
            return ToolResponseMessage(
                {'error': "L'ID utilisateur est requis."},
                tool_call.data['id']
            )
            
        # On vérifie que l'utilisateur existe
        user = self.bot.get_user(user_id)
        if user is None:
            return ToolResponseMessage(
                {'error': "Utilisateur non trouvé."},
                tool_call.data['id']
            )
            
        header = f"Consultation des infos de ***{user.name}***"
        if context.fetch_author().id == user.id:
            header = "Consultation de vos infos"
            
        # On récupère les infos personnalisées
        infos = self.get_user_custom(user)
        if infos is None:
            return ToolResponseMessage(
                {'error': "Aucune information personnalisée trouvée pour cet utilisateur."},
                tool_call.data['id'],
                header=header
            )
        
        # S'assurer que infos est bien une chaîne
        if not isinstance(infos, str):
            infos = str(infos)
            
        return ToolResponseMessage(
            {'user': f'{user.name} (ID:{user.id})', 'infos': infos},
            tool_call.data['id'],
            header=header
        )
        
    def _tool_update_user_info(self, tool_call: ToolCall, context: MessageGroup) -> ToolResponseMessage:
        """Outil pour mettre à jour les informations utilisateur."""
        user_id = tool_call.arguments.get('user_id')
        infos = tool_call.arguments.get('infos')
        if user_id is None or infos is None:
            return ToolResponseMessage(
                {'error': "L'ID utilisateur et les informations sont requis."},
                tool_call.data['id']
            )
        # On vérifie que l'utilisateur existe
        user = self.bot.get_user(user_id)
        if user is None:
            return ToolResponseMessage(
                {'error': "Utilisateur non trouvé."},
                tool_call.data['id']
            )
            
        # On vérifie que l'utilisateur dont on modifie les infos est bien celui qui a appelé l'outil
        if context.fetch_author().id != user.id:
            return ToolResponseMessage(
                {'error': f"L'ID du demandeur ({context.fetch_author().id}) doit correspondre à l'ID de l'utilisateur dont on modifie les infos ({user.id})."},
                tool_call.data['id']
            )
            
        # On met à jour les infos personnalisées
        current = self.get_user_custom(user)
        if len(infos) > 500:
            return ToolResponseMessage(
                {'error': "Les informations ne doivent pas dépasser 500 caractères.", 'old_content': current or ''},
                tool_call.data['id']
            )
        
        if current is None:
            self.update_user_custom(user, infos)
            return ToolResponseMessage(
                {'old_content': '', 'new_content': infos},
                tool_call.data['id'],
                header=f"Mise à jour des infos de ***{user.name}***"
            )
        else:
            self.update_user_custom(user, infos)
            return ToolResponseMessage(
                {'old_content': current, 'new_content': infos},
                tool_call.data['id'],
                header=f"Mise à jour de vos infos"
            )
        
    def _tool_math_eval(self, tool_call: ToolCall, context: MessageGroup) -> ToolResponseMessage:
        expression = tool_call.arguments.get('expression')
        if not expression:
            return ToolResponseMessage({'error': 'Aucune expression mathématique fournie.'}, tool_call.data['id'])
    
        try:
            result = float(ne.evaluate(expression))
            if result.is_integer():
                result = int(result)
            return ToolResponseMessage({'result': result}, tool_call.data['id'], 
                                       header=f"Calcul de `{expression}`")
        except Exception as e:
            return ToolResponseMessage({'error': str(e)}, tool_call.data['id'])
        
    # Détection de réponse ----------------------
    
    async def detect_response(self, message: discord.Message) -> bool:
        """Détecte si le message est une réponse à un chatbot."""
        if message.author.bot:
            return False
        
        mode = self.get_guild_config(message.guild, 'chatbot_mode', str)
        if mode == 'off': # Chatbot désactivé
            return False
        
        if self.bot.user.mentioned_in(message): # Modes 'strict', 'opportunistic' et 'greedy'
            return True
        
        s = re.search(rf'\b{re.escape(self.bot.user.name.lower())}\b', message.content.lower())
        if s and mode in ['opportunistic', 'greedy']:
            # On vérifie le score d'opportunité si en mode opportuniste
            if mode == 'opportunistic':
                score = await self._opportunistic_agent.score_message(message)
                threshold = self.get_guild_config(message.guild, 'opportunist_threshold', int)
                if score < threshold:
                    return False
            return True
        
        return False
    
    # TRANSCRIPTION CONTEXTUELLE AUDIO ----------------------
    
    async def transcript_audio_callback(self, interaction: Interaction, message: discord.Message):
        if interaction.channel_id != message.channel.id:
            return await interaction.response.send_message("**Action impossible** × Le message doit être dans le même salon", ephemeral=True, delete_after=10)
        if not message.attachments:
            return await interaction.response.send_message("**Erreur** × Aucun fichier n'est attaché au message.", ephemeral=True, delete_after=10)
        
        session = await self.get_channel_chat_session(interaction.channel)
        
        file = await session.extract_audio(message)
        if not file:
            return await interaction.response.send_message(content="**Erreur** × Aucun fichier audio ou vidéo valide n'a été trouvé dans le message.", ephemeral=True, delete_after=10)
        
        # Modal
        modal = TranscriptPrompt()
        await interaction.response.send_modal(modal)
        await modal.wait()
        if modal.is_finished():
            return await interaction.followup.send(content="**Action annulée** × La modal a été fermée.", ephemeral=True)
        
        prompt = modal.audioprompt.value.strip()
        if not prompt:
            prompt = ""
        
        try:
            transcript = await session.get_audio_transcript(file, prompt=prompt, return_type='text')
        except Exception as e:
            logger.error(f"Erreur lors de la transcription audio : {e}")
            if isinstance(file, io.BytesIO):
                file.close()
            elif isinstance(file, Path):
                file.unlink()
            return await interaction.followup.send(content=f"**Erreur** × La transcription n'a pas pu être générée : {e}", ephemeral=True)
        except OpenAIError as e:
            return await interaction.followup.send(content=f"**Erreur** × La transcription n'a pas pu être générée : {e}", ephemeral=True)
        
        if not transcript:
            return await interaction.followup.send(content="**Erreur** × La transcription est vide ou n'a pas pu être générée.", ephemeral=True)
        
        if type(file) is Path:
            file.unlink()
            
        await interaction.followup.send(content="**Transcription terminée** × La transcription a été générée avec succès.", ephemeral=True)
        
        transcript = f">>> {transcript}\n-# Transcription demandée par {interaction.user.mention}"
        
        content = []
        if len(transcript) >= 2000:
            content = [transcript[i:i+2000] for i in range(0, len(transcript), 2000)]
        else:
            content = [transcript]
        for _, chunk in enumerate(content):
            await message.reply(chunk, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
            
        await interaction.delete_original_response()
    
    # EVENTS ===============================================
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not isinstance(message.channel, VALID_CHATBOT_CHANNELS):
            return
        
        if message.author.bot:
            return
        
        channel = message.channel
        if not await self.detect_response(message):
            return
        
        # Ajouter immédiatement l'ID pour éviter les doublons en cas d'édition rapide
        self._processed_messages.append(message.id)

        async with channel.typing():
            session = await self.get_channel_chat_session(channel)
            group = await session.append_message(message)
            if group is None:
                return
            resp = await session.get_answer()
                
            tools : list[ToolResponseMessage] = group.get_messages(lambda m: isinstance(m, ToolResponseMessage))
            
            # Traitement des icons et des headers
            tools_repr = []
            lines = list(set([trm.tool_repr for trm in tools if trm.tool_repr]))
            lines.reverse()  # On inverse pour afficher les plus récents en premier
            if lines:
                tools_repr = '\n-# ' + '\n-# '.join(lines[::-1]) + '\n'
                resp = tools_repr + resp
            
            # On coupe le message en morceaux de 2000 caractères si nécessaire
            while len(resp) > 2000:
                part = resp[:2000]
                resp = resp[2000:]
                ans_msg = await message.reply(part, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
            if resp:
                ans_msg = await message.reply(resp, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
            
            group.last_completion.message = ans_msg
            
    @commands.Cog.listener()
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        if not isinstance(after.channel, VALID_CHATBOT_CHANNELS):
            return
        
        if after.author.bot:
            return
        
        if after.id in self._processed_messages:
            # Si le message a déjà été traité, on ignore l'édition
            return
        
        if after.created_at < datetime.now(timezone.utc) - MAX_EDITION_AGE:
            # Si le message est trop vieux, on ignore l'édition
            return
        
        channel = after.channel
        if not await self.detect_response(after):
            return
        
        # Ajouter immédiatement l'ID pour éviter les doublons
        self._processed_messages.append(after.id)

        async with channel.typing():
            session = await self.get_channel_chat_session(channel)
            group = await session.append_message(after)
            if group is None:
                return
            
            resp = await session.get_answer()
                
            tools : list[ToolResponseMessage] = group.get_messages(lambda m: isinstance(m, ToolResponseMessage))
            
            # Traitement des icons et des headers
            tools_repr = []
            lines = list(set([trm.tool_repr for trm in tools if trm.tool_repr]))
            lines.reverse()  # On inverse pour afficher les plus récents en premier
            if lines:
                tools_repr = '\n-# ' + '\n-# '.join(lines[::-1]) + '\n'
                resp = tools_repr + resp

            # On coupe le message en morceaux de 2000 caractères si nécessaire
            while len(resp) > 2000:
                part = resp[:2000]
                resp = resp[2000:]
                ans_msg = await after.reply(part, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
            if resp:
                ans_msg = await after.reply(resp, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
                
            group.last_completion.message = ans_msg
            
    # COMMANDES =====================================================
    
    @app_commands.command(name='info')
    async def cmd_info(self, interaction: Interaction):
        """Affiche des informations sur le bot et son utilisation."""
        bot_name = self.bot.user.name
        if not interaction.guild:
            return await interaction.response.send_message(f"**{bot_name}** n'est pas configuré pour fonctionner en DM. Veuillez l'inviter sur un serveur Discord pour l'utiliser.", ephemeral=True)
        bot_color = interaction.guild.me.color
        session = await self.get_channel_chat_session(interaction.channel)
        config = self.get_full_guild_config(interaction.guild)
        desc = """*Une assistante intelligente pour Discord conçue pour répondre à des questions, effectuer des recherches sur le web ou encore analyser des vidéos et des images.*"""
        embed = discord.Embed(title=bot_name, description=desc, color=bot_color)
        
        session_info = f"Taille du contexte : `{sum(g.total_token_count for g in session.agent._history_groups)} tokens`\n"
        session_info += f"Nombre d'intéractions : `{len(session.agent._history_groups)}`"
        embed.add_field(name=f"Session sur {interaction.channel.name}", value=session_info, inline=True)
        
        params_info = f"Mode de réponse : `{config['chatbot_mode'].upper()}`\n"
        if config['chatbot_mode'] == 'opportunistic':
            params_info += f"Seuil d'opportunité : `{config['opportunist_threshold']}%`"
        embed.add_field(name="Paramètres globaux", value=params_info, inline=True)
        embed.set_thumbnail(url=self.bot.user.display_avatar.url)
        embed.set_footer(text=f"Utilisez /chatbot pour configurer MARIA")
        await interaction.response.send_message(embed=embed)
        
    @app_commands.command(name='memoire')
    async def cmd_memory(self, interaction: Interaction):
        """Afficher et modifier vos informations personnelles communiquées à MARIA."""
        user_info = self.get_user_custom(interaction.user)
        modal = UserInfoModal()
        if user_info:
            modal.userinfo.default = user_info
        await interaction.response.send_modal(modal)
        await modal.wait()
        if modal.is_finished():
            if modal.userinfo.value:
                self.update_user_custom(interaction.user, modal.userinfo.value)
                await interaction.followup.send(f"**Vos préférences ont été mises à jour** ⸱ Voici ce que vous avez communiqué :\n```{modal.userinfo.value}```", ephemeral=True)
            else:
                self.remove_user_custom(interaction.user)
                await interaction.followup.send("**Vos préférences ont été supprimées**", ephemeral=True)
                
    chatbot_settings = app_commands.Group(name='chatbot', description="Paramètres globaux du chatbot", default_permissions=discord.Permissions(manage_messages=True), guild_only=True)
    
    @chatbot_settings.command(name='forget')
    async def forget(self, interaction: Interaction):
        """Supprime l'historique de conversation interne du chatbot."""
        if interaction.channel.id not in self._SESSIONS:
            return await interaction.response.send_message("**Aucune session de chat en cours**", ephemeral=True)
        
        session = self._SESSIONS[interaction.channel.id]
        session.agent.clear_history()
        await interaction.response.send_message("**Historique de conversation interne supprimé**", ephemeral=True)

    @chatbot_settings.command(name='mode')
    @app_commands.choices(
        mode=[
            app_commands.Choice(name='Désactivé', value='off'),
            app_commands.Choice(name='Mentions directes uniquement', value='strict'),
            app_commands.Choice(name='Mentions directes + indirectes (si pertinentes)', value='opportunistic'),
            app_commands.Choice(name='Mentions directes + indirectes (sans condition)', value='greedy')
        ]
    )
    async def set_chatbot_mode(self, interaction: Interaction, mode: Literal['off', 'strict', 'opportunistic', 'greedy']):
        """Configure le mode de réponse du chatbot pour le serveur
        
        :param mode: Le mode de réponse du chatbot
        """
        self.set_guild_config(interaction.guild, 'chatbot_mode', mode)
        session = self._SESSIONS.get(interaction.channel.id)
        if session:
            session.answer_modes = mode
        if mode == 'off':
            if interaction.channel.id in self._SESSIONS:
                await self.remove_channel_chat_session(interaction.channel)
            return await interaction.response.send_message("**Le chatbot est désactivé pour ce canal**", ephemeral=True)
        await interaction.response.send_message(f"**Mode du chatbot modifié** ⸱ {mode.upper()}\n-# Le seuil de pertinence de réponse est réglable avec `/chatbot opp_threshold` pour le mode 'OPPORTUNISTE'.", ephemeral=True)
        
    @chatbot_settings.command(name='opp_threshold')
    async def set_opportunistic_threshold(self, interaction: Interaction, threshold: app_commands.Range[int, 0, 100] = 50):
        """Configure le seuil d'opportunité au dessus duquel le chatbot répond aux mentions indirectes (en mode 'OPPORTUNISTE')
        
        :param threshold: Le seuil d'opportunité en pourcentage (0-100), par défaut 50%
        """
        self.set_guild_config(interaction.guild, 'opportunist_threshold', threshold)
        await interaction.response.send_message(f"**Seuil d'opportunité modifié** ⸱ `{threshold}%`\n-# Le chatbot répondra aux mentions indirectes ayant un score d'opportunité supérieur à cette valeur (si le mode est activé).", ephemeral=True)
    
    
    
    # COMMANDES SPECIALES ------------------------------------------------
    
    @commands.command(name='refreshstatus')
    @commands.is_owner()
    async def force_refresh_status(self, ctx: commands.Context):
        """Force la mise à jour du statut du bot."""
        try:
            new_status = await self._status_updater_agent.get_status()
            await self.bot.change_presence(activity=discord.Activity(type=discord.ActivityType.custom, name='custom', state=new_status))
            await ctx.send(f"**Statut mis à jour** ⸱ `{new_status}`\n-# /!\ Sera remplacé par le statut généré automatiquement à la prochaine mise à jour programmée.", ephemeral=True)
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du statut : {e}")
            await ctx.send(f"**Erreur** × {e}", ephemeral=True)
            
    
    
async def setup(bot):
    await bot.add_cog(Chat(bot))