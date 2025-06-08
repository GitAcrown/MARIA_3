import io
import logging
import os
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal
import numexpr as ne

import discord
from discord import Interaction, app_commands
from discord.ext import commands, tasks
from moviepy import VideoFileClip
from openai import AsyncOpenAI

from common import dataio
from common.llm.agents import *
from common.llm.classes import *

logger = logging.getLogger(f'MARIA3.{__name__.split(".")[-1]}')

# CONSTANTES ----------------------------------------------------------------

DEVELOPER_PROMPT_TEMPLATE = lambda current_datetime, weekday: f"""[BEHAVIOR]
Tu es MARIA, un assistant intelligent conversant avec plusieurs utilisateurs dans un salon textuel Discord.
Ne mets jamais ni ton nom ni ton identifiant dans tes réponses, tu n'inclues pas de balises dans tes réponses.
Tu peux analyser les images qu'on te donne avec précision et détail lorsqu'on te le demande.
Les informations contenues dans les balises '<...>' dans les messages de l'historique sont des metadonnées extraites de pièces jointes qui peuvent t'aider à répondre.
Tu peux éventuellement mentionner un utilisateur en mettant son ID de cette manière : <@user.id>. Ne le fais qu'en cas d'absolue nécessité.

[INFO]
- Format des messages utilisateurs: `[<message.id>] <author.name> (<author.id>) : <message.content>`
- Date actuelle (ISO 8601): {current_datetime}
- Jour de la semaine: {weekday}

[TOOLS]
- CALCULS MATHÉMATIQUES: Tu peux évaluer des expressions mathématiques complexes avec précision. Utilise cet outil pour tous calculs, conversions ou résolutions d'équations.
- RECHERCHE DE RÉSUMÉS: Tu peux rechercher des informations dans des résumés de conversations passées. Utilise cet outil pour trouver des informations sur des sujets spécifiques ou des conversations passées.
- NAVIGATION WEB: Tu peux effectuer des recherches internet et naviguer sur des pages web (par blocs). Utilise cet outil pour trouver des informations sur des sujets spécifiques ou des conversations passées.

[RESPONSE GUIDELINES]
- Sois la plus directe et concise possible dans tes réponses, va droit au but;
- Suis scrupuleusement les demandes des utilisateurs;
- Explique ton raisonnement pour les calculs complexes;
- Utilise les outils de manière proactive pour enrichir tes réponses;
- N'hésite pas à poser des questions de clarification si nécessaire.
"""

STATUS_UPDATE_INTERVAL = 30 #MINUTES
ACTIVITY_MAX_MESSAGES = 10
SUMMARY_MAX_AGE = timedelta(days=30)

# PARAMÈTRES =================================================================

ANSWER_MODES = {
    'NONE': "Ne réponds à aucun message (désactive le bot).",
    'LAZY': "Répondre uniquement aux mentions directes.",
    'GREEDY': "Répondre aux messages contenant des mentions directes et indirectes.",
    'AUTO': "Détermine automatiquement la nécessité de répondre en fonction du contexte.",
}

# CLASSES ===================================================================

class Summary:
    def __init__(self, channel_id: int, start_time: datetime, end_time: datetime, authors: list[int], summary: str):
        self.channel_id = channel_id
        self.start_time = start_time
        self.end_time = end_time
        self.authors = authors
        self.summary = summary

    def to_enriched_dict(self, guild: discord.Guild) -> dict:
        return {
            'channel_id': self.channel_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'authors': [f'{guild.get_member(author_id).name} ({author_id})' for author_id in self.authors],
            'summary': self.summary
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Summary':
        return cls(
            channel_id=data['channel_id'],
            start_time=datetime.now().fromisoformat(data['start_time']),
            end_time=datetime.now().fromisoformat(data['end_time']),
            authors=[int(author) for author in data['authors'].split(',')],
            summary=data['summary']
        )
    
    # Helpers
    def contains_words(self, words: list[str]) -> bool:
        """Vérifie si le résumé contient les mots donnés."""
        return any(word in self.summary.lower() for word in words)

    def is_mentionned(self, member: discord.Member) -> bool:
        """Vérifie si un membre est mentionné dans le résumé textuel (ID entre [])"""
        return f"[{member.id}]" in self.summary

    def retrieve_authors(self) -> list[discord.Member]:
        """Récupère les auteurs des messages."""
        return [self.guild.get_member(author_id) for author_id in self.authors]
    
class GuildChatSession:
    def __init__(self,
                 cog: 'Main',
                 guild: discord.Guild, 
                 developer_prompt_template: Callable[[None], str] = DEVELOPER_PROMPT_TEMPLATE,
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
                 answer_mode: Literal['NONE', 'LAZY', 'GREEDY', 'AUTO'] = 'AUTO'):
        self._cog = cog
        self.guild = guild

        self.developer_prompt_template = developer_prompt_template

        # Paramètres de l'agent
        self.agent = ChatbotAgent(
            client=cog._gptclient,
            developer_prompt_template=self._get_developer_prompt,
            temperature=temperature,
            completion_model=completion_model,
            transcription_model=transcription_model,
            max_completion_tokens=max_completion_tokens,
            context_window=context_window,
            max_context_age=max_context_age,
            tools=tools,
            tools_enabled=tools_enabled,
            tools_parallel_calls=tools_parallel_calls
        )

        # Paramètres de la session
        self.answer_mode = answer_mode

    def _get_developer_prompt(self) -> str:
        return self.developer_prompt_template(datetime.now(), datetime.now().strftime('%A'))

    # Complétion texte
    async def append_user_message(self, message: discord.Message) -> MessageGroup:
        """Ajoute un message utilisateur à la session."""
        if message.author.bot:
            return
        
        if message.reference and message.reference.resolved:
            ref_message = UserMessage.from_discord_message(message.reference.resolved)
            ctx_message = UserMessage.from_discord_message(message)
            ctx_message.add_components(MetadataTextComponent('REFERENCE', message_id=message.reference.resolved.id))
            return self.agent.create_and_insert_group(ref_message, ctx_message)
        else:
            ctx_message = UserMessage.from_discord_message(message)
            return self.agent.create_and_insert_group(ctx_message)
    
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
                path = self.data.get_subfolder('temp', create=True) / f'{datetime.now().timestamp()}.mp4'
                await attachment.save(path)
                clip = VideoFileClip(str(path))
                audio = clip.audio
                if not audio:
                    return None
                audio_path = path.with_suffix('.wav')
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
        raise ValueError(f"Type de retour invalide: {return_type}")
    
class StatusUpdaterAgent:
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.model = "gpt-4.1-nano"
        self.system_prompt = """
        Invente un seul petit message de statut mignon, marrant et original pour un bot Discord sans emojis. Il doit être très court, concis (quelques mots max.) et en français.
        Il peut être en rapport avec le bot et son activité (une assistante et chatbot IA). Tu te genre au féminin.
        N'hésite pas à utiliser des références internet drôles. 

        Exemples du style de statut :
        - "Navigue sur le web"
        - "Flotte entre les channels"
        - "Veille sur les serveurs"
        - "Chill sur le net"
        - "En mode chatbot"

        Tu dois répondre dans un format JSON avec la clé 'status' et la valeur du statut.
        """

    class MessageStatus(BaseModel):
        status: str

    async def get_status(self) -> str:
        """Récupère un statut en fonction de l'humeur."""
        prompt = self.system_prompt
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "developer", "content": prompt}],
            response_format=self.MessageStatus
        )
        if not response.choices[0].message.parsed:
            raise Exception("Erreur lors de la récupération du statut.")
        return response.choices[0].message.parsed.status

# COG =======================================================================
class Main(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = dataio.get_instance(self)

        # BDD
        guild_config = dataio.DictTableBuilder(
            name='guild_config',
            default_values={
                'answer_mode': 'AUTO',
                'attention_span': 45, # En secondes
                'activity_threshold': 10, # Nombre de messages par minute
                'enable_summary': True
            },
            insert_on_reconnect=True
        )
        guild_summary = dataio.TableBuilder(
            """CREATE TABLE IF NOT EXISTS guild_summary (
                channel_id INTEGER,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                authors TEXT,
                summary TEXT,
                PRIMARY KEY (start_time, end_time)
            )"""
        )
        self.data.map_builders(discord.Guild, guild_config, guild_summary)

        # Agents
        self._gptclient = AsyncOpenAI(api_key=self.bot.config['OPENAI_API_KEY'])
        self._status_updater_agent = StatusUpdaterAgent(self._gptclient)
        self._monitor_agent = MonitorAgent(self._gptclient)
        self._guilds_sessions : dict[int, GuildChatSession] = {}
        self._summary_agents : dict[int, SummaryAgent] = {}

        self._monitor_attention : dict[int, dict] = {}
        self._channel_activity : dict[int, list[discord.Message]] = {}

        self.update_status.start()

        # Outils
        self.AGENT_TOOLS = [
            Tool(
                name='math_eval',
                description="Évalue une expression mathématique. Utilise la syntaxe Python standard avec les opérateurs mathématiques classiques.",
                properties={
                    'expression': {
                        'type': 'string',
                        'description': "L'expression mathématique à évaluer"
                    }
                },
                function=self._tool_math_eval
            ),
            Tool(
                name='search_summaries',
                description="Recherche des résumés de conversations passées. Il faut fournir au moins un critère de recherche (mot-clés, auteurs, date approximative).",
                properties={
                    'keywords': {
                        'type': 'string',
                        'description': "Les mots-clés à rechercher dans les résumés (séparés par des virgules)"
                    },
                    'author_ids': {
                        'type': 'string',
                        'description': "Les ID des auteurs à rechercher dans les résumés (séparés par des virgules)"
                    },
                    'approx_time': {
                        'type': 'string',
                        'description': "La date approximative à rechercher dans les résumés (format: YYYY-MM-DD)"
                    }
                },
                function=self._tool_search_summaries
            )
        ]

    async def cog_unload(self):
        self.update_status.cancel()
        await self._gptclient.close()
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

    # Config =================================================================

    def get_guild_config(self, guild: discord.Guild) -> dict:
        """Récupère la configuration d'un serveur."""
        return self.data.get(guild).get_dict_values('guild_config')
    
    def set_guild_config(self, guild: discord.Guild, **config):
        """Met à jour la configuration d'un serveur."""
        self.data.get(guild).set_dict_values('guild_config', config)

    # Guild Chatbots =================================================================

    async def get_guild_chat_session(self, guild: discord.Guild) -> GuildChatSession:
        """Récupère la session de chat d'un serveur."""
        if guild.id not in self._guilds_sessions:
            self.retrieve_tools()
            self._guilds_sessions[guild.id] = GuildChatSession(self, guild, tools=self.AGENT_TOOLS)
        return self._guilds_sessions[guild.id]

    async def remove_guild_chat_session(self, guild: discord.Guild) -> None:
        """Supprime la session de chat d'un serveur."""
        if guild.id in self._guilds_sessions:
            del self._guilds_sessions[guild.id]

    # Custom Chatbots =================================================================
    # TODO

    # Outils ===================================================================

    def retrieve_tools(self):
        for cog in self.bot.cogs.values():
            if cog.qualified_name == self.qualified_name:
                continue
            if hasattr(cog, 'AGENT_TOOLS'):
                for tool in cog.AGENT_TOOLS:
                    if tool.name not in (t.name for t in self.AGENT_TOOLS):
                        self.AGENT_TOOLS.append(tool)
                        logger.info(f"i --- Outil '{tool.name}' ajouté depuis '{cog.qualified_name}'")

    # Calcul mathématique
    def _tool_math_eval(self, tool_call: ToolCall, context: MessageGroup) -> ToolResponseMessage:
        expression = tool_call.arguments.get('expression')
        if not expression:
            return ToolResponseMessage({'error': 'Aucune expression mathématique fournie.'}, tool_call.data['id'])
    
        try:
            result = float(ne.evaluate(expression))
            if result.is_integer():
                result = int(result)
            return ToolResponseMessage({'result': result}, tool_call.data['id'], header=f"**Calcul de l'expression** `{expression}`")
        except Exception as e:
            return ToolResponseMessage({'error': str(e)}, tool_call.data['id'])
        
    # Recherche de résumés
    def _tool_search_summaries(self, tool_call: ToolCall, context: MessageGroup) -> ToolResponseMessage:
        keywords = tool_call.arguments.get('keywords', '').split(',')
        author_ids = [int(author_id) for author_id in tool_call.arguments.get('author_ids', '').split(',')]
        approx_time = tool_call.arguments.get('approx_time', '')
        if approx_time:
            approx_time = datetime.strptime(approx_time, '%Y-%m-%d')
        else:
            approx_time = None

        guild = context.fetch_guild()
        if not keywords and not author_ids and not approx_time:
            return ToolResponseMessage({'error': 'Il faut fournir au moins un critère de recherche (mot-clés, auteurs, date approximative)'}, tool_call.data['id'])
        authors = [guild.get_member(author_id) for author_id in author_ids]
        summaries = self.search_summary(guild, keywords=keywords, authors=authors, approx_time=approx_time)
        return ToolResponseMessage({'summaries': [s.to_enriched_dict(guild) for s in summaries]}, tool_call.data['id'], header="**Consultation des discussions passées**")

    # Response Type Detection ==================================================

    def get_channel_activity(self, channel: discord.TextChannel | discord.Thread) -> list[discord.Message]:
        """Récupère l'activité d'un salon."""
        if channel.id not in self._channel_activity:
            self._channel_activity[channel.id] = []
        return self._channel_activity[channel.id]
    
    def update_channel_activity(self, message: discord.Message) -> None:
        """Met à jour l'activité d'un salon."""
        if message.author.bot:
            return
        activity = self.get_channel_activity(message.channel)
        activity.append(message)
        activity = activity[-ACTIVITY_MAX_MESSAGES:] # On garde les [ACTIVITY_MAX_MESSAGES] derniers messages

    def score_channel_activity(self, channel: discord.TextChannel | discord.Thread) -> float:
        """Calcule le score d'activité d'un salon déterminé à partir de la densité de messages par minute."""
        activity = self.get_channel_activity(channel)
        if not activity:
            return 0
        # On fait la moyenne des messages par minute
        return sum([(datetime.now(timezone.utc) - message.created_at).total_seconds() for message in activity]) / len(activity)
    
    def get_response_type(self, message: discord.Message) -> Literal['reply', 'send']:
        """Détermine si le bot doit utiliser message.reply ou channel.send pour répondre à un message."""
        config = self.get_guild_config(message.guild)
        if self.score_channel_activity(message.channel) > int(config['activity_threshold']):
            return 'reply' # Si y'a trop de messages par minute, on répond en réponse
        return 'send' # Sinon, on répond en message direct

    # Monitor =================================================================

    def get_attention(self, channel: discord.TextChannel) -> dict:
        """Récupère l'attention du bot sur un salon."""
        if channel.id not in self._monitor_attention:
            self._monitor_attention[channel.id] = {
                'last_user_id': 0,
                'last_use_timestamp': 0
            }
        return self._monitor_attention[channel.id]

    def register_attention(self, message: discord.Message) -> None:
        """Enregistre l'attention du bot sur un salon."""
        channel = message.channel
        attention = self.get_attention(channel)
        attention['last_user_id'] = message.author.id
        attention['last_use_timestamp'] = message.created_at.replace(tzinfo=timezone.utc)

    def check_attention(self, message: discord.Message) -> bool:
        """Vérifie si un message est dans l'attention du bot sur un salon."""
        channel = message.channel
        config = self.get_guild_config(message.guild)
        attention = self.get_attention(channel)
        if message.author.id == attention['last_user_id'] and attention['last_use_timestamp'] > message.created_at - timedelta(seconds=int(config['attention_span'])):
            return True
        return False

    async def detect_reply(self, bot: discord.Client, message: discord.Message) -> bool:
        """Détermine si il faut répondre à un message à partir du mode de réponse du serveur."""
        config = self.get_guild_config(message.guild)
        
        if config['answer_mode'] == 'NONE':
            return False
        
        if bot.user.mentioned_in(message): # Modes LAZY, GREEDY et AUTO = On répond si le bot est mentionné directement
            if message.mention_everyone:
                return False
            return True
        
        if config['answer_mode'] == 'GREEDY': # SEULEMENT mode GREEDY = On répond aussi si le bot est mentionné indirectement (mention de nom ou display_name en regex)
            s = re.search(rf'\b{re.escape(bot.user.name.lower())}\b', message.content.lower())
            if s:
                return True
            s = re.search(rf'\b{re.escape(bot.user.display_name.lower())}\b', message.content.lower())
            if s:
                return True
            return False
        
        if config['answer_mode'] == 'AUTO': # SEULMENT mode AUTO = On répond si le bot est dans l'attention du bot et que Monitor décide de répondre
            if self.check_attention(message):
                resp = await self._monitor_agent.detect_message_reply(self.bot.user, message)
                return resp.choice == 'YES'
        return False

    # Summary =================================================================

    def get_summary_agent(self, channel: discord.TextChannel) -> SummaryAgent:
        """Récupère l'agent de résumé d'un salon."""
        if channel.id not in self._summary_agents:
            self._summary_agents[channel.id] = SummaryAgent(self._gptclient)
        return self._summary_agents[channel.id]
    
    def remove_summary_agent(self, channel: discord.TextChannel) -> None:
        """Supprime l'agent de résumé d'un salon."""
        if channel.id in self._summary_agents:
            del self._summary_agents[channel.id]

    def create_summary(self, channel: discord.TextChannel, start_time: datetime, end_time: datetime, authors: list[discord.Member], summary: str) -> None:
        """Crée un résumé pour une fenêtre de temps donnée."""
        guild = channel.guild
        self.data.get(guild).execute(
            "INSERT INTO guild_summary (channel_id, start_time, end_time, authors, summary) VALUES (?, ?, ?, ?, ?)",
            channel.id, start_time.isoformat(), end_time.isoformat(), ','.join([str(author.id) for author in tuple(set(authors))]), summary
        )

    def update_summary(self, channel: discord.TextChannel, start_time: datetime, end_time: datetime, authors: list[discord.Member], summary: str) -> None:
        """Met à jour un résumé existant."""
        guild = channel.guild
        self.data.get(guild).execute(
            "UPDATE guild_summary SET authors = ?, summary = ? WHERE channel_id = ? AND start_time = ? AND end_time = ?",
            ','.join([str(author.id) for author in authors]), summary, channel.id, start_time.isoformat(), end_time.isoformat()
        )

    def delete_old_summaries(self, channel: discord.TextChannel) -> None:
        """Supprime les résumés d'un salon qui sont trop vieux."""
        guild = channel.guild
        summaries = self.get_all_summary(guild, channel)
        for summary in summaries:
            if summary.end_time < datetime.now(timezone.utc) - SUMMARY_MAX_AGE:
                self.data.get(guild).execute(
                    "DELETE FROM guild_summary WHERE channel_id = ? AND start_time = ? AND end_time = ?",
                    channel.id, summary.start_time.isoformat(), summary.end_time.isoformat()
                )
                logger.info(f"i --- Résumé supprimé : {summary.start_time} {summary.end_time}")
                break

    async def handle_summarization(self, message: discord.Message, type: Literal['user', 'assistant', 'bot']) -> SummaryAgent.AgentSummary | None:
        """Gère la summarization d'un message."""
        guild = message.guild
        config = self.get_guild_config(guild)
        if not bool(config['enable_summary']):
            return None
        
        agent = self.get_summary_agent(message.channel)
        try_summarize = False
        if type == 'user':
            agent.add_user_message(message)
            try_summarize = True
        elif type == 'assistant':
            agent.add_assistant_message(message, is_self=True)
        elif type == 'bot':
            agent.add_assistant_message(message)
        else:
            raise ValueError(f"Type de message invalide: {type}")
        
        if try_summarize:
            agentsummary = await agent.maybe_summarize()
            if agentsummary:
                self.create_summary(message.channel, agentsummary.start_time, agentsummary.end_time, agentsummary.authors, agentsummary.text)
            return agentsummary
        return None

    def get_all_summary(self, guild: discord.Guild, channel: discord.TextChannel | discord.Thread | None = None) -> list[Summary]:
        """Récupère le résumé d'un salon."""
        if channel:
            r = self.data.get(guild).fetchall("SELECT * FROM guild_summary WHERE channel_id = ?", (channel.id,))
        else:
            r = self.data.get(guild).fetchall("SELECT * FROM guild_summary")
        return [Summary.from_dict(s) for s in r]
    
    def search_summary(self, 
                       guild: discord.Guild,
                       *,
                       channel: discord.TextChannel | discord.Thread | None = None,
                       authors: list[discord.Member] | None = None,
                       keywords: list[str] | None = None,
                       approx_time: datetime | None = None) -> list[Summary]:
        """Recherche un résumé en croisant différents critères (exclusif)"""
        self.delete_old_summaries(channel)
        summaries = self.get_all_summary(guild, channel)
        logger.warning(f"Recherche de résumés : {authors} {keywords} {approx_time}")
        if authors:
            summaries = [s for s in summaries if any(author.id in s.authors for author in authors)]
        if keywords:
            summaries = [s for s in summaries if any(keyword in s.summary for keyword in keywords)]
        if approx_time:
            summaries = [s for s in summaries if approx_time.strftime('%Y-%m-%d') in s.start_time.strftime('%Y-%m-%d')]
        logger.warning(f"Résumés trouvés : {len(summaries)}")
        return summaries
    
    # Events ==================================================================

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not isinstance(message.channel, (discord.TextChannel, discord.Thread)):
            return

        if message.author.bot:
            await self.handle_summarization(message, type='bot')
            return
        
        channel = message.channel
        self.update_channel_activity(message)
        await self.handle_summarization(message, type='user')
        if await self.detect_reply(self.bot, message):
            async with channel.typing():
                session = await self.get_guild_chat_session(channel.guild)
                group = await session.append_user_message(message)
                response = await session.get_answer()
                self.register_attention(message)

                headers = []
                if group.search_for_message_components(MetadataTextComponent, lambda c: 'AUDIO' in c.data['text']):
                    headers.append("Transcription audio")

                tools : list[ToolResponseMessage] = group.get_messages(lambda m: isinstance(m, ToolResponseMessage))
                headers.extend(list(set([trm.header for trm in tools if trm.header])))
                if headers:
                    response = '\n-# ' + '\n-# '.join(headers[::-1]) + '\n' + response

                response_type = self.get_response_type(message)
                if len(response) > 2000:
                    response = response[:1997] + '...'
                
                if response_type == 'reply':
                    message = await message.reply(response, mention_author=False)
                else:
                    message = await message.channel.send(response, mention_author=False)
                await self.handle_summarization(message, type='assistant')
                group.last_completion.message = message

    @commands.Cog.listener()
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        config = self.get_guild_config(before.guild)
        if not bool(config['enable_summary']):
            return
        
        if before.content != after.content:
            summ_agent = self.get_summary_agent(after.channel)
            if summ_agent.try_removing_message(before):
                summ_agent.add_user_message(after)

            # Redétection de demande de réponse
            if await self.detect_reply(self.bot, after):
                async with after.channel.typing():
                    session = await self.get_guild_chat_session(after.guild)
                    group = await session.append_user_message(after)
                    response = await session.get_answer()
                    self.register_attention(after)

                    headers = []
                    if group.search_for_message_components(MetadataTextComponent, lambda c: 'AUDIO' in c.data['text']):
                        headers.append("Transcription audio")

                    tools : list[ToolResponseMessage] = group.get_messages(lambda m: isinstance(m, ToolResponseMessage))
                    headers.extend(list(set([trm.header for trm in tools if trm.header])))
                    if headers:
                        response = '\n-# ' + '\n-# '.join(headers[::-1]) + '\n' + response

                    if len(response) > 2000:
                        response = response[:1997] + '...'

                    last_message = after.channel.last_message
                    if last_message == after:
                        message = await after.channel.send(response, mention_author=False)
                    else:
                        message = await after.reply(response, mention_author=False)
                    await self.handle_summarization(message, type='assistant')
                    group.last_completion.message = message

    @commands.Cog.listener()
    async def on_message_delete(self, message: discord.Message):
        config = self.get_guild_config(message.guild)
        if not bool(config['enable_summary']):
            return
        
        summ_agent = self.get_summary_agent(message.channel)
        summ_agent.try_removing_message(message)

    # COMMANDES >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    @app_commands.command(name='info')
    async def info(self, interaction: Interaction):
        """Affiche des informations sur le bot et son utilisation."""
        bot_name = self.bot.user.name
        bot_color = self.bot.user.color
        session = await self.get_guild_chat_session(interaction.guild)
        config = self.get_guild_config(interaction.guild)
        embed = discord.Embed(title=bot_name, description="Un assistant intelligent pour Discord", color=bot_color)
        embed.add_field(name="Taille du contexte", value=f"{sum(len(g.total_token_count) for g in session.agent._context)} tokens", inline=True)
        embed.add_field(name="Durée d'attention", value=f"{config['attention_span']} secondes", inline=True)
        embed.add_field(name="Mode de réponse", value=f"{ANSWER_MODES[config['answer_mode']]}", inline=True)
        embed.add_field(name="Résumé automatique", value=f"{'Activé' if config['enable_summary'] else 'Désactivé'}", inline=True)
        embed.set_thumbnail(url=self.bot.user.display_avatar.url)
        embed.set_footer(text=f"Version {self.bot.version} • Utilisez /settings pour configurer le bot.")
        await interaction.response.send_message(embed=embed)
        
    settings_group = app_commands.Group(name='settings', description="Paramètres généraux", default_permissions=discord.Permissions(manage_messages=True))

    @settings_group.command(name='answer-mode')
    @app_commands.choices(mode=[app_commands.Choice(name=f'{desc} ({value})', value=value) for value, desc in ANSWER_MODES.items()])
    async def answer_mode(self, interaction: Interaction, mode: Literal['NONE', 'LAZY', 'GREEDY', 'AUTO']):
        """Définit le mode de réponse du bot.
        
        :param mode: Mode de réponse
        """
        self.set_guild_config(interaction.guild, answer_mode=mode)
        await interaction.response.send_message(f"**MODE DE RÉPONSE MODIFIÉ** ⸱ *{ANSWER_MODES[mode]}*", ephemeral=True)

    @settings_group.command(name='attention-span')
    async def attention_span(self, interaction: Interaction, span: app_commands.Range[int, 10, 300]):
        """Définit la durée d'attention du bot sur un salon lorsque le mode de réponse est AUTO.
        
        :param span: Durée d'attention en secondes
        """
        self.set_guild_config(interaction.guild, attention_span=span)
        await interaction.response.send_message(f"**DURÉE D'ATTENTION MODIFIÉE** ⸱ L'IA gardera son attention sur le dernier utilisateur pendant {span} secondes.", ephemeral=True)

    @settings_group.command(name='activity-threshold')
    async def activity_threshold(self, interaction: Interaction, threshold: app_commands.Range[int, 1, 100]):
        """Définit le seuil d'activité d'un salon nécessaire pour que le bot ait besoin de répondre en réponse directe.
        
        :param threshold: Seuil d'activité en messages par minute
        """
        self.set_guild_config(interaction.guild, activity_threshold=threshold)
        await interaction.response.send_message(f"**SEUIL D'ACTIVITÉ MODIFIÉ** ⸱ Réglé sur *{threshold}* messages par minute", ephemeral=True)

    @settings_group.command(name='enable-summary')
    async def enable_summary(self, interaction: Interaction, toggle: bool):
        """Autorise ou non le bot à effectuer des résumés automatiques des conversations afin d'enrichir sa mémoire.
        
        :param toggle: True pour activer, False pour désactiver
        """
        self.set_guild_config(interaction.guild, enable_summary=toggle)
        if toggle:
            await interaction.response.send_message(f"**CAPACITÉ DE RÉSUMÉ** ⸱ Le bot effectuera des résumés automatiques des conversations afin d'enrichir sa mémoire.", ephemeral=True)
        else:
            await interaction.response.send_message(f"**CAPACITÉ DE RÉSUMÉ** ⸱ Le bot ne fera plus de résumés automatiques des conversations.", ephemeral=True)

async def setup(bot):
    await bot.add_cog(Main(bot))
