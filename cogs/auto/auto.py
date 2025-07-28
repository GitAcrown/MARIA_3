import io
import asyncio
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
from discord.ext import commands, tasks
from moviepy import VideoFileClip
from openai import AsyncOpenAI

from common import dataio
from common.llm.agents import *
from common.llm.classes import *

logger = logging.getLogger(f'MARIA3.{__name__.split(".")[-1]}')

PROPOSAL_EMOJI = '<:suggestion:1399507101814096004>'
FALLBACK_EMOJI = '💬'  # Emoji de fallback si l'emoji personnalisé n'est pas disponible
PROPOSAL_TYPES = Literal['audio_transcription']

# CLASSES ----------------------------------------------------

class GuildBotSession:
    def __init__(self, 
                 cog: 'Auto',
                 agent: ChatbotAgent):
        self.cog = cog
        self.agent = agent
    
    # Transcription audio
    async def fetch_message_audio(self, message: discord.Message) -> io.BytesIO | Path | None:
        """Extrait le texte d'un message audio (vocal discord seulement)."""
        for attachment in message.attachments:
            # Message audio Discord
            if attachment.content_type and attachment.content_type.startswith('audio'):
                buffer = io.BytesIO()
                buffer.name = attachment.filename
                await attachment.save(buffer, seek_begin=True)
                return buffer
        return None
    
    async def get_transcription(self, message: discord.Message) -> str:
        """Transcrit le contenu audio d'un message."""
        audio = await self.fetch_message_audio(message)
        if audio is None:
            return ""
        # Vérifie si le fichier audio est un fichier vocal Discord
        transcription = await self.agent.extract_audio_transcript(audio)
        if transcription is None:
            return ""
        # Nettoie la transcription
        transcription = transcription.strip()
        return transcription

# COG ----------------------------------------------------
class Auto(commands.Cog):
    """Cog pour les fonctionnalités de suggestions automatiques de l'IA."""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = dataio.get_instance(self)
        
        guild_settings = dataio.DictTableBuilder(
            name='guild_settings',
            default_values={
                'proposal_expiration': 180,
                'suggest_audio_transcription': True,
            }
        )
        self.data.map_builders(discord.Guild, guild_settings)
        
        self._gptclient = AsyncOpenAI(
            api_key=self.bot.config['OPENAI_API_KEY']
        )
        
        self._SESSIONS : dict[int, GuildBotSession] = {}
        self._proposals : dict[int, dict[Literal[PROPOSAL_TYPES], bool]] = {}
        
    # SESSION MANAGEMENT ------------------------------------
        
    async def get_guild_agent(self, guild: discord.Guild) -> GuildBotSession:
        """Recupère ou crée une session pour le bot dans la guilde."""
        if guild.id not in self._SESSIONS:
            agent = ChatbotAgent(
                client=self._gptclient,
                developer_prompt=""
            )
            self._SESSIONS[guild.id] = GuildBotSession(cog=self, agent=agent)
        return self._SESSIONS[guild.id]
    
    async def remove_guild_agent(self, guild: discord.Guild) -> None:
        """Supprime la session du bot dans la guilde."""
        if guild.id in self._SESSIONS:
            del self._SESSIONS[guild.id]
            
    # CONFIGURATION MANAGEMENT ------------------------------------
    
    def get_guild_config(self, guild: discord.Guild, key: str):
        """Récupère la configuration d'une guilde."""
        return self.data.get(guild).get_dict_value('guild_settings', key)
    
    def set_guild_config(self, guild: discord.Guild, key: str, value: Union[str, int, bool]) -> None:
        """Met à jour la configuration d'une guilde."""
        self.data.get(guild).set_dict_value('guild_settings', key, value)
        
            
    # PROPOSALS MANAGEMENT ------------------------------------
    
    def add_proposal(self, message: discord.Message, proposal_type: PROPOSAL_TYPES) -> None:
        """Ajoute une proposition de type audio transcription."""
        if message.id not in self._proposals:
            self._proposals[message.id] = {}
        if proposal_type not in self._proposals[message.id]:
            self._proposals[message.id][proposal_type] = False
            
    def remove_proposal(self, message: discord.Message, proposal_type: PROPOSAL_TYPES) -> None:
        """Supprime une proposition de type audio transcription."""
        if message.id in self._proposals and proposal_type in self._proposals[message.id]:
            del self._proposals[message.id][proposal_type]
            if not self._proposals[message.id]:
                del self._proposals[message.id]
                
    def has_proposal(self, message: discord.Message, proposal_type: PROPOSAL_TYPES) -> bool:
        """Vérifie si une proposition de type audio transcription existe pour le message."""
        return message.id in self._proposals and proposal_type in self._proposals[message.id]
    
    def set_proposal_status(self, message: discord.Message, proposal_type: PROPOSAL_TYPES, status: bool) -> None:
        """Met à jour le statut d'une proposition de type audio transcription."""
        if message.id not in self._proposals:
            self._proposals[message.id] = {}
        self._proposals[message.id][proposal_type] = status
        
    def get_proposal_status(self, message: discord.Message, proposal_type: PROPOSAL_TYPES) -> bool:
        """Récupère le statut d'une proposition de type audio transcription."""
        return self._proposals.get(message.id, {}).get(proposal_type, False)
        
    async def _schedule_proposal_expiration(self, message: discord.Message, expiration: int):
        """Programme l'expiration d'une proposition après un délai donné."""
        await asyncio.sleep(expiration)
        if self.has_proposal(message, 'audio_transcription') and not self.get_proposal_status(message, 'audio_transcription'):
            # Si la proposition n'a pas été traitée, on supprime la réaction
            try:
                emoji = self.get_proposal_emoji(message.guild)
                await message.clear_reaction(emoji)
            except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                pass  # Ignore les erreurs de permissions ou si le message n'existe plus
            self.remove_proposal(message, 'audio_transcription')
            
    def get_proposal_emoji(self, guild: discord.Guild) -> str:
        """Récupère l'emoji de proposition, avec fallback si l'emoji personnalisé n'est pas disponible."""
        try:
            # Tente d'extraire l'ID de l'emoji personnalisé
            if PROPOSAL_EMOJI.startswith('<:') and PROPOSAL_EMOJI.endswith('>'):
                emoji_id = int(PROPOSAL_EMOJI.split(':')[2][:-1])
                emoji = discord.utils.get(self.bot.emojis, id=emoji_id)
                if emoji and emoji.is_usable():
                    return str(emoji)
            # Si l'emoji personnalisé n'est pas disponible, utilise le fallback
            logger.warning(f"Emoji personnalisé {PROPOSAL_EMOJI} non disponible, utilisation du fallback")
            return FALLBACK_EMOJI
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'emoji: {e}")
            return FALLBACK_EMOJI
        
    # LISTENER ----------------------------------------------------
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Écoute les messages pour déclencher des actions automatiques."""
        if message.author.bot or not message.guild:
            return
        
        # Ignore les messages dans les salons privés
        if isinstance(message.channel, discord.DMChannel):
            return
        
        attachments = message.attachments
        if attachments:
            if any(attachment.content_type and attachment.content_type.startswith('audio') for attachment in attachments):
                # Si le message contient un fichier audio, on transcrit
                logger.debug(f"Fichier audio détecté dans le message {message.id}")
                if bool(self.get_guild_config(message.guild, 'suggest_audio_transcription')):
                    expiration = int(self.get_guild_config(message.guild, 'proposal_expiration'))
                    self.add_proposal(message, 'audio_transcription')
                    logger.info(f"Proposition audio créée pour le message {message.id}")
                    try:
                        emoji = self.get_proposal_emoji(message.guild)
                        await message.add_reaction(emoji)
                        logger.debug(f"Emoji {emoji} ajouté au message {message.id}")
                        # Programme l'expiration de la proposition en arrière-plan
                        asyncio.create_task(self._schedule_proposal_expiration(message, expiration))
                    except (discord.Forbidden, discord.HTTPException) as e:
                        logger.warning(f"Impossible d'ajouter une réaction: {e}")
                        self.remove_proposal(message, 'audio_transcription')
                else:
                    logger.debug("Suggestions de transcription audio désactivées")
        else:
            # Si le message n'a pas d'attachement, on vérifie les propositions
            if self.has_proposal(message, 'audio_transcription'):
                # Si une proposition audio transcription existe, on la supprime
                try:
                    emoji = self.get_proposal_emoji(message.guild)
                    await message.clear_reaction(emoji)
                except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                    pass  # Ignore les erreurs de permissions
                self.remove_proposal(message, 'audio_transcription')
                    
    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        """Écoute les réactions pour traiter les propositions."""
        logger.debug(f"Réaction ajoutée: {reaction.emoji} par {user.name}")
        
        if user.bot:
            logger.debug("Utilisateur bot ignoré")
            return
        
        if reaction.emoji != PROPOSAL_EMOJI and str(reaction.emoji) != FALLBACK_EMOJI:
            logger.debug(f"Emoji différent de {PROPOSAL_EMOJI} ou {FALLBACK_EMOJI}, ignoré")
            return
        
        message = reaction.message
        logger.debug(f"Vérification de la proposition pour le message {message.id}")
        
        if not self.has_proposal(message, 'audio_transcription'):
            logger.debug("Aucune proposition trouvée pour ce message")
            return
        
        # Marque la proposition comme étant traitée pour éviter les doubles exécutions
        if self.get_proposal_status(message, 'audio_transcription'):
            logger.debug("Proposition déjà en cours de traitement")
            return
        
        logger.info(f"Début de la transcription audio pour le message {message.id}")
        self.set_proposal_status(message, 'audio_transcription', True)
        
        try:
            async with message.channel.typing():
                # Récupère la session de la guilde
                session = await self.get_guild_agent(message.guild)
                
                # Traite la transcription audio
                transcription = await session.get_transcription(message)
                if transcription:
                    content = f">>> {transcription}\n-# Transcription demandée par {user.mention}"
                    await message.reply(content, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
                else:
                    await message.reply("**Aucune transcription disponible pour ce message audio.**", mention_author=False, allowed_mentions=discord.AllowedMentions.none())
                
        except Exception as e:
            logger.error(f"Erreur lors de la transcription audio: {e}")
            await message.reply("**Erreur lors de la transcription audio.**", mention_author=False, allowed_mentions=discord.AllowedMentions.none())
        
        finally:
            # Supprime la proposition et la réaction à la fin
            try:
                emoji = self.get_proposal_emoji(message.guild)
                await message.clear_reaction(emoji)
            except discord.Forbidden:
                # Le bot n'a pas les permissions pour supprimer les réactions
                logger.warning(f"Permissions insuffisantes pour supprimer les réactions dans {message.guild.name}")
            except discord.NotFound:
                # Le message ou la réaction n'existe plus
                pass
            except Exception as e:
                logger.error(f"Erreur lors de la suppression de la réaction: {e}")
            
            self.remove_proposal(message, 'audio_transcription')
            logger.info(f"Transcription terminée pour le message {message.id}")
    # COMMANDS ----------------------------------------------------
    
    auto_group = app_commands.Group(name='auto', description="Paramètres des fonctionnalités automatiques de l'IA", default_permissions=discord.Permissions(manage_messages=True))
    
    @auto_group.command(name='proposal_expiration', description="Modifie la durée de validité des propositions")
    async def proposal_expiration(self, interaction: Interaction, duration: app_commands.Range[int, 30, 600]):
        """Modifie la durée de validité des propositions."""
        self.set_guild_config(interaction.guild, 'proposal_expiration', duration)
        await interaction.response.send_message(f"**Durée de validité des propositions modifiée** ⸱ `{duration} secondes`", ephemeral=True)
    
    @auto_group.command(name='transcription', description="Active ou désactive la suggestion de transcription audio")
    async def transcription(self, interaction: Interaction, status: bool):
        """Active ou désactive la suggestion de transcription audio."""
        self.set_guild_config(interaction.guild, 'suggest_audio_transcription', status)
        if status:
            await interaction.response.send_message("**Suggestions de transcription audio activées**", ephemeral=True)
        else:
            await interaction.response.send_message("**Suggestions de transcription audio désactivées**", ephemeral=True)

async def setup(bot):
    await bot.add_cog(Auto(bot))
