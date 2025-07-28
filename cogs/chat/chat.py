import io
import logging
import os
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Union
import numexpr as ne

import discord
from discord import Interaction, app_commands
from discord.ext import commands
from moviepy import VideoFileClip
from openai import AsyncOpenAI

from common import dataio
from common.llm.agents import *
from common.llm.classes import *

logger = logging.getLogger(f'MARIA3.{__name__.split(".")[-1]}')

# CONSTANTES ----------------------------------------------------

DEVELOPER_PROMPT_TEMPLATE = lambda args: f"""Tu es MARIA, assistante IA conversant sur Discord.

RÈGLES:
- Adopter le ton de la conversation
- Concis et direct
- Utilise le Markdown Discord si utile
- Si tu affiches plusieurs liens, mets-les entre crochets <lien>
- Mentionne les utilisateurs avec <@user.id> seulement si nécessaire

CONTEXTE:
- Messages de l'historique: `pseudo (user_id) : contenu` (ne formate pas tes propres messages de la sorte)
- Pièces jointes représentées par des métadonnées en <> (ex: <AUDIO: filename.wav>)
- Date actuelle: {args['weekday']} {args['datetime']}

OUTILS DISPONIBLES:
- Infos utilisateur: informations personnelles et préférences de l'utilisateur
- Calculs mathématiques: expressions complexes et conversions
- Recherche web: informations actuelles via Google

Si tu ne sais pas, recherche sur internet. Si toujours aucune réponse, dis-le clairement."""

# PARAMETRES -----------------------------------------------------

VALID_CHATBOT_CHANNELS = Union[discord.TextChannel, discord.VoiceChannel, discord.Thread]
MAX_EDITION_AGE = timedelta(minutes=2)  # Age maximal des messages pour l'édition

# Répertoire temporaire
TEMP_DIR = Path('./temp')
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

# UI ========================================================================

class UserInfoModal(discord.ui.Modal, title="Préférences et infos"):
    def __init__(self) -> None:
        super().__init__(timeout=None)
        self.userinfo = discord.ui.TextInput(
            label="Vos préférences personnelles",
            style=discord.TextStyle.long,
            placeholder="Informations ou préférences à partager avec le chatbot",
            min_length=0,
            max_length=500
        )
        self.add_item(self.userinfo)
        
    async def on_submit(self, interaction: Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        return self.stop()
        
    async def on_error(self, interaction: Interaction, error: Exception) -> None:
        return await interaction.response.send_message(f"**Erreur** × {error}", ephemeral=True)

# CLASSES -----------------------------------------------------

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
        """Extrait le texte d'un message audio."""
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
    
    async def get_audio_transcript(self, file: io.BytesIO | Path, return_type: Literal['text', 'component'] = 'text') -> str | MetadataTextComponent:
        """Récupère le texte d'un message audio."""
        transcript = await self.agent.extract_audio_transcript(file)
        if return_type == 'text':
            return transcript
        elif return_type == 'component':
            return MetadataTextComponent('AUDIO', filename=file.name, transcript=transcript)

class Chat(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = dataio.get_instance(self)
        
        # CONFIG
        guild_config = dataio.DictTableBuilder(
            name='guild_config',
            default_values={
                'chatbot_mode': 'opportunistic',
                'opportunist_threshold': 50  # Seuil de score pour le mode 'opportunistic'
            }
        )
        self.data.map_builders(discord.Guild, guild_config)
        
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
        
        # Sessions de chat
        self._SESSIONS : dict[int, ChannelChatSession] = {}
        
        # Messages déjà traités
        self._processed_messages = set()  # Pour éviter les doublons
        
        # Outils
        self.GLOBAL_TOOLS = [
            Tool(
                name='user_info',
                description='Consulter les informations personnelles et préférences de l’utilisateur.',
                properties={
                    'user_id': {
                        'type': 'integer',
                        'description': "L'ID de l'utilisateur"
                    }
                },
                function=self._tool_user_info
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
        await self._gptclient.close()
        self.data.close_all()
        
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
                'weekday': datetime.now(timezone.utc).strftime('%A'),
                'datetime': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
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
                        
    def _tool_user_info(self, tool_call: ToolCall, context: MessageGroup) -> ToolResponseMessage:
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
            
        # On récupère les infos personnalisées
        infos = self.get_user_custom(user)
        if infos is None:
            return ToolResponseMessage(
                {'error': "Aucune information personnalisée trouvée pour cet utilisateur."},
                tool_call.data['id']
            )
        
        # S'assurer que infos est bien une chaîne
        if not isinstance(infos, str):
            infos = str(infos)
            
        return ToolResponseMessage(
            {'user': f'{user.name} (ID:{user.id})', 'infos': infos},
            tool_call.data['id'],
            header=f"[+] Préférences de ***{user.name}***"
        )
        
    def _tool_math_eval(self, tool_call: ToolCall, context: MessageGroup) -> ToolResponseMessage:
        expression = tool_call.arguments.get('expression')
        if not expression:
            return ToolResponseMessage({'error': 'Aucune expression mathématique fournie.'}, tool_call.data['id'])
    
        try:
            result = float(ne.evaluate(expression))
            if result.is_integer():
                result = int(result)
            return ToolResponseMessage({'result': result}, tool_call.data['id'], header=f"Calcul de `{expression}`")
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
                print(f"Score d'opportunité pour {message.id}: {score}")
                threshold = self.get_guild_config(message.guild, 'opportunist_threshold', int)
                if score < threshold:
                    return False
            return True
        
        return False
    
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
        
        async with channel.typing():
            session = await self.get_channel_chat_session(channel)
            group = await session.append_message(message)
            if group is None:
                return
            resp = await session.get_answer()
            headers = []
            if group.search_for_message_components(lambda c: isinstance(c, MetadataTextComponent) and 'AUDIO' in c.data['text']):
                headers.append("[+] Contenu audio")
            if group.search_for_message_components(lambda c: isinstance(c, MetadataTextComponent) and 'VIDEO' in c.data['text']):
                headers.append("[+] Contenu vidéo")
                
            tools : list[ToolResponseMessage] = group.get_messages(lambda m: isinstance(m, ToolResponseMessage))
            headers.extend(list(set([trm.header for trm in tools if trm.header])))
            if headers:
                resp = '\n-# ' + '\n-# '.join(headers[::-1]) + '\n' + resp

            # On coupe le message en morceaux de 2000 caractères si nécessaire
            while len(resp) > 2000:
                part = resp[:2000]
                resp = resp[2000:]
                ans_msg = await message.reply(part, mention_author=False)
            if resp:
                ans_msg = await message.reply(resp, mention_author=False)
            
            group.last_completion.message = ans_msg
            self._processed_messages.add(message.id)
            
        if len(self._processed_messages) > 100:
            self._processed_messages = set(list(self._processed_messages)[-100:])
            
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
        
        async with channel.typing():
            session = await self.get_channel_chat_session(channel)
            group = await session.append_message(after)
            if group is None:
                return
            
            resp = await session.get_answer()
            headers = []
            if group.search_for_message_components(lambda c: isinstance(c, MetadataTextComponent) and 'AUDIO' in c.data['text']):
                headers.append("[+] Contenu audio")
            if group.search_for_message_components(lambda c: isinstance(c, MetadataTextComponent) and 'VIDEO' in c.data['text']):
                headers.append("[+] Contenu vidéo")
                
            tools : list[ToolResponseMessage] = group.get_messages(lambda m: isinstance(m, ToolResponseMessage))
            headers.extend(list(set([trm.header for trm in tools if trm.header])))
            if headers:
                resp = '\n-# ' + '\n-# '.join(headers[::-1]) + '\n' + resp

            # On coupe le message en morceaux de 2000 caractères si nécessaire
            while len(resp) > 2000:
                part = resp[:2000]
                resp = resp[2000:]
                ans_msg = await after.reply(part, mention_author=False)
            if resp:
                ans_msg = await after.reply(resp, mention_author=False)
                
            group.last_completion.message = ans_msg
            self._processed_messages.add(after.id)
            
    # COMMANDES =====================================================
    
    @app_commands.command(name='info')
    async def info(self, interaction: Interaction):
        """Affiche des informations sur le bot et son utilisation."""
        bot_name = self.bot.user.name
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
        embed.set_footer(text=f"Utilisez /settings pour configurer le chatbot")
        await interaction.response.send_message(embed=embed)
        
    @app_commands.command(name='preferences')
    async def preferences(self, interaction: Interaction):
        """Afficher ou modifier vos préférences communiquées au chatbot."""
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
                
    @app_commands.command(name='forget')
    async def forget(self, interaction: Interaction):
        """Supprime l'historique de conversation interne du chatbot."""
        if interaction.channel.id not in self._SESSIONS:
            return await interaction.response.send_message("**Aucune session de chat en cours**", ephemeral=True)
        
        session = self._SESSIONS[interaction.channel.id]
        session.agent.clear_history()
        await interaction.response.send_message("**Historique de conversation interne supprimé**", ephemeral=True)
                
    settings_group = app_commands.Group(name='settings', description="Paramètres globaux du chatbot", default_permissions=discord.Permissions(manage_messages=True))

    @settings_group.command(name='mode')
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
        await interaction.response.send_message(f"**Mode du chatbot modifié** ⸱ {mode.upper()}\n-# Le seuil de pertinence de réponse est réglable avec `/settings opp_threshold` pour le mode 'OPPORTUNISTE'.", ephemeral=True)
        
    @settings_group.command(name='opp_threshold')
    async def set_opportunistic_threshold(self, interaction: Interaction, threshold: app_commands.Range[int, 0, 100] = 50):
        """Configure le seuil d'opportunité au dessus duquel le chatbot répond aux mentions indirectes (en mode 'OPPORTUNISTE')
        
        :param threshold: Le seuil d'opportunité en pourcentage (0-100), par défaut 50%
        """
        self.set_guild_config(interaction.guild, 'opportunist_threshold', threshold)
        await interaction.response.send_message(f"**Seuil d'opportunité modifié** ⸱ `{threshold}%`\n-# Le chatbot répondra aux mentions indirectes ayant un score d'opportunité supérieur à cette valeur (si le mode est activé).", ephemeral=True)
    
async def setup(bot):
    await bot.add_cog(Chat(bot))