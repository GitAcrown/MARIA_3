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

NATIVE_EMOJIS = {
    'audio_transcription': '<:transcript_audio:1399501329985962114>'
}
PROPOSAL_TYPES = Literal['audio_transcription']
PROPOSAL_EXPIRATION = 300  # 5 minutes

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
            
    # LISTENER ----------------------------------------------------
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Écoute les messages pour déclencher des actions automatiques."""
        if message.author.bot or not message.guild:
            return
        
        # Ignore les messages dans les salons privés
        if isinstance(message.channel, discord.DMChannel):
            return
        
        # Récupère la session de la guilde
        session = await self.get_guild_agent(message.guild)
        
        attachments = message.attachments
        if attachments:
            if any(attachment.content_type and attachment.content_type.startswith('audio') for attachment in attachments):
                # Si le message contient un fichier audio, on transcrit
                if self.data.get_guild_config(message.guild, 'suggest_audio_transcription', True):
                    self.add_proposal(message, 'audio_transcription')
                    await message.add_reaction(NATIVE_EMOJIS['audio_transcription'])
                    await asyncio.sleep(PROPOSAL_EXPIRATION)
                    
    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        """Écoute les réactions pour traiter les propositions."""
        if user.bot or reaction.emoji != NATIVE_EMOJIS['audio_transcription']:
            return
        
        message = reaction.message
        if not self.has_proposal(message, 'audio_transcription'):
            return
        
        # Récupère la session de la guilde
        session = await self.get_guild_agent(message.guild)
        
        # Traite la transcription audio
        transcription = await session.get_transcription(message)
        if transcription:
            content = f">>> {transcription}\n-# Transcription demandée par {user.mention}"
            await message.reply(content, mention_author=False, allowed_mentions=discord.AllowedMentions.none())

async def setup(bot):
    await bot.add_cog(Auto(bot))
